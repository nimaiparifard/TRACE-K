"""Post-training graph compression and paired evaluation for a residual R-GCN."""

import hashlib
import json
import platform
from collections import defaultdict
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .graph import Activations, collate, graph
from .probe import Probe
from .train import paired_accuracy


METHODS = ("random_node", "layer_importance", "paraphrase_selection", "spectral",
           "similarity", "signal_aware", "prediction_aware")
REDUCTIONS = (.25, .50, .75, .875)


def adjacency(edge, count):
    matrix = edge.new_zeros((count, count))
    source, dest = edge[:, :2].long().T
    matrix.index_put_((source, dest), edge[:, 3], accumulate=True)
    return (matrix + matrix.T) / 2


def quotient(edge, assignment, count):
    source, dest, relation = edge[:, :3].long().T
    ids = (assignment[source] * count + assignment[dest]) * 3 + relation
    keys, inverse = ids.unique(return_inverse=True)
    weight = edge.new_zeros(len(keys)).index_add_(0, inverse, edge[:, 3])
    return torch.stack((keys // (count * 3), keys // 3 % count,
                        keys % 3, weight), dim=1)


def aggregate(values, assignment, count):
    totals = values.new_zeros((count, values.shape[1])).index_add_(0, assignment, values)
    return totals / assignment.bincount(minlength=count).clamp_min(1)[:, None]


def select_centers(features, maximum, priority=None):
    distances = torch.cdist(features, features).square().clamp_min_(0)
    matrix = distances.numpy()
    if priority is None:
        priority = np.ones(len(features), dtype=matrix.dtype)
    centers = []
    nearest = np.full(len(features), np.inf, dtype=matrix.dtype)
    scores = matrix.mean(axis=1) * priority
    for _ in range(maximum):
        center = scores.argmax().item()
        centers.append(center)
        nearest = np.minimum(nearest, matrix[center])
        scores = nearest * priority
        scores[centers] = -1
    return torch.tensor(centers), distances


def compact_graph(x, edge, anchors, centers, distances, merge):
    assignment = distances[:, centers].argmin(dim=1)
    assignment[centers] = torch.arange(len(centers))
    if merge:
        return (aggregate(x, assignment, len(centers)),
                quotient(edge, assignment, len(centers)), assignment[anchors]), assignment
    retained = torch.full((len(x),), -1, dtype=torch.long)
    retained[centers] = torch.arange(len(centers))
    endpoints = retained[edge[:, :2].long()]
    mask = (endpoints >= 0).all(dim=1)
    compact_edge = torch.cat((endpoints[mask], edge[mask, 2:]), dim=1)
    return (x[centers], compact_edge, assignment[anchors]), assignment


def prepare_batch(graph_data, device, needs_grad=False):
    x, edge, anchors = graph_data
    batch = collate([(x, edge, 0)], device)
    return (batch[0].requires_grad_(needs_grad), *batch[1:-1], anchors.to(device))


def readout(model, batch, final):
    inputs, pooled = model.vocabulary_input(batch, final, return_pool=True, shared_graph=True)
    logits = F.linear(inputs, model.unembedding, model.bias)
    return inputs, pooled, logits


def salience(model, graph_data, final, top, device):
    with torch.enable_grad():
        batch = prepare_batch(graph_data, device, needs_grad=True)
        inputs = model.vocabulary_input(batch, final, shared_graph=True)
        direction = model.unembedding[top[:, 0]] - model.unembedding[top[:, 1]]
        gradient, = torch.autograd.grad((inputs * direction).sum(), batch[0])
    return gradient.norm(dim=1).cpu()


def spectral_basis(edge, count):
    matrix = adjacency(edge, count)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix.sum(1).diag() - matrix)
    positive = eigenvalues > 1e-4
    return eigenvectors[:, positive][:, :8].contiguous(), eigenvalues[positive][:8]


def spectral_error(compact_edge, assignment, basis, eigenvalues, count):
    modes = aggregate(basis, assignment, count)
    source, dest = compact_edge[:, :2].long().T
    energy = ((modes[source] - modes[dest]).square() * compact_edge[:, 3, None]).sum(0) / 2
    return ((energy / eigenvalues - 1).square()).mean().item() if eigenvalues.numel() else 0.


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measured_probe(model, graph_data, final, device, profile):
    if not profile:
        return readout(model, prepare_batch(graph_data, device), final), {}
    sync(device)
    started = perf_counter()
    batch = prepare_batch(graph_data, device)
    sync(device)
    packed = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        allocated = torch.cuda.memory_allocated(device)
    values = readout(model, batch, final)
    sync(device)
    return values, dict(batch_ms=(packed - started) * 1000,
                        inference_ms=(perf_counter() - packed) * 1000,
                        gpu_forward_extra_bytes=(torch.cuda.max_memory_allocated(device) - allocated
                                                 if device.type == "cuda" else None),
                        gpu_probe_peak_bytes=(torch.cuda.max_memory_allocated(device)
                                              if device.type == "cuda" else None))


def records(row, logits):
    gold = row["gold"]
    predictions = logits.argmax(1).tolist()
    matches = logits[:, gold:gold + 1]
    rank = 1 + (logits > matches).sum(1) + (logits[:, :gold] == matches).sum(1)
    return prompt_records(row, predictions, rank.tolist())


def prompt_records(row, predictions, ranks):
    gold = row["gold"]
    return [dict(fact_id=row["fact_id"], gold=gold, base=base, prediction=prediction, rank=position)
            for base, prediction, position in zip(row["base"], predictions, ranks, strict=True)]


def accuracy(rows):
    correct = [row["prediction"] == row["gold"] for row in rows]
    failed = [hit for hit, row in zip(correct, rows) if row["base"] != row["gold"]]
    return dict(accuracy=float(np.mean(correct)), mrr=float(np.mean([1 / row["rank"] for row in rows])),
                recovery_rate=float(np.mean(failed)) if failed else None)


def mean_metrics(rows):
    result = {key: (float(np.mean([row[key] for row in rows])) if rows[0][key] is not None else None)
              for key in rows[0]}
    if "inference_ms" in result:
        result["inference_p95_ms"] = float(np.percentile([row["inference_ms"] for row in rows], 95))
        for key in ("gpu_forward_extra_bytes", "gpu_probe_peak_bytes"):
            result[f"max_{key}"] = max(row[key] for row in rows) if result[key] is not None else None
    if "observed_certificate_violations" in result:
        result["observed_certificate_violations"] = sum(row["observed_certificate_violations"] for row in rows)
    return result


def full_graph(item, options):
    x, edge, _ = graph(item, 0, options)
    views, depth = item[1]["states"].shape[:2]
    return x, edge, torch.arange(depth - 1, views * depth, depth)


def ablation_graph(item, options, name):
    views, depth = item[1]["states"].shape[:2]
    if isinstance(name, int):
        x, edge, _ = graph(item, 0, options, excluded_layer=name)
    else:
        x, edge, _ = graph(item, 0, options | name)
    retained_depth = len(x) // views
    return x, edge, torch.arange(retained_depth - 1, views * retained_depth, retained_depth)


@torch.no_grad()
def layer_scores(model, dataset, options, device, limit, seed):
    scores = defaultdict(list)
    selection = np.random.default_rng(seed).permutation(len(dataset))[:limit]
    for index in selection:
        item = dataset[int(index)]
        final = item[1]["states"][:, -1].to(device, dtype=torch.float32)
        full, _ = measured_probe(model, full_graph(item, options), final, device, False)
        teacher = full[-1].softmax(-1)
        depth = item[1]["states"].shape[1]
        for layer in range(depth):
            ablated, _ = measured_probe(model, ablation_graph(item, options, layer), final, device, False)
            scores[layer].append(F.kl_div(ablated[-1].log_softmax(-1), teacher,
                                          reduction="batchmean").item())
    return {layer: float(np.mean(values)) for layer, values in scores.items()}


def configuration_name(method, reduction, mix):
    return f"{method}/{reduction:g}/{mix:g}"


@torch.no_grad()
def evaluate_ablations(model, dataset, options, device, reference, seed, edge_only=False):
    depth = dataset[0][1]["states"].shape[1]
    variants = {"no_cross_view": {"cross_view": False},
                "no_inter_layer": {"inter_layer": False},
                "degree_preserving_rewire": {"random_edges": True}}
    if not edge_only:
        variants.update(last_layer_only={"layers": "last"}, middle_layers_only={"layers": "middle"})
        variants.update({f"leave_out_layer_{layer}": layer for layer in range(depth)})
    results = {name: [] for name in variants}
    for index, item in enumerate(dataset, start=1):
        row, saved = item
        final = saved["states"][:, -1].to(device, dtype=torch.float32)
        inputs = []
        for name, variant in variants.items():
            graph_data = ablation_graph(item, options, variant)
            inputs.append(model.vocabulary_input(prepare_batch(graph_data, device), final, shared_graph=True))
        gold = row["gold"]
        names = list(variants)
        for start in range(0, len(inputs), 8):
            logits = F.linear(torch.stack(inputs[start:start + 8]), model.unembedding, model.bias)
            matches = logits[..., gold:gold + 1]
            ranks = 1 + (logits > matches).sum(-1) + (logits[..., :gold] == matches).sum(-1)
            for name, predictions, positions in zip(names[start:start + 8], logits.argmax(-1).tolist(),
                                                    ranks.tolist(), strict=True):
                results[name].extend(prompt_records(row, predictions, positions))
        if index % 100 == 0:
            print(f"Ablations: {index}/{len(dataset)} facts", flush=True)
    return {name: accuracy(rows) | paired_accuracy(rows, reference, np.random.default_rng(seed))
            for name, rows in results.items()}


@torch.no_grad()
def evaluate_compact(model, dataset, options, configs, importance, device, seed, profile):
    metrics, predictions = defaultdict(list), defaultdict(list)
    output_norm = torch.linalg.matrix_norm(model.output.weight, ord=2).item()
    head_row_norm = model.unembedding.norm(dim=1).max().item()
    basis_cache = {}
    for index, item in enumerate(dataset, start=1):
        row, saved = item
        started = perf_counter()
        x, edge, anchors = full_graph(item, options)
        build_ms = (perf_counter() - started) * 1000
        views, depth = saved["states"].shape[:2]
        final = saved["states"][:, -1].to(device, dtype=torch.float32)
        started = perf_counter()
        (full_input, full_pool, full_logits), full_cost = measured_probe(
            model, (x, edge, anchors), final, device, profile)
        full_ready = perf_counter() - started
        if profile:
            metrics["full"].append(dict(nodes=len(x), edges=len(edge),
                                        graph_bytes=tensor_bytes(x) + tensor_bytes(edge) + tensor_bytes(anchors),
                                        final_bytes=tensor_bytes(saved["states"][:, -1]),
                                        graph_build_ms=build_ms, **full_cost))
            predictions["full"].extend(records(row, full_logits))
        full_probs = full_logits.softmax(-1)
        top_values, top = full_logits.topk(2)
        margin = top_values[:, 0] - top_values[:, 1]
        shape = views, depth
        if shape not in basis_cache:
            basis_cache[shape] = spectral_basis(edge, len(x))
        basis, eigenvalues = basis_cache[shape]
        start = perf_counter()
        unit = F.normalize(x, dim=1)
        geometry = F.normalize(basis.T, dim=1).T * (len(x) / max(basis.shape[1], 1)) ** .5
        signal_time = perf_counter() - start
        plans = {}
        prior = None
        salience_time = 0.
        validation_inputs = []
        validation_results = []
        for method, reduction, mix in configs:
            count = min(len(x) - 1, max(1, round(len(x) * (1 - reduction))))
            key = method, mix
            if key not in plans:
                started = perf_counter()
                merge = method in {"spectral", "similarity", "signal_aware", "prediction_aware"}
                priority = None
                extra_time = 0.
                if method == "spectral":
                    features = geometry
                elif method in {"signal_aware", "prediction_aware"}:
                    features = torch.cat((unit, mix ** .5 * geometry), dim=1)
                else:
                    features = unit
                if method == "prediction_aware":
                    if prior is None:
                        salience_started = perf_counter()
                        prior = salience(model, (x, edge, anchors), final, top, device)
                        prior = 1 + prior / prior.mean().clamp_min(1e-8)
                        salience_time = perf_counter() - salience_started
                        started += salience_time
                    features = torch.cat((features, F.normalize(prior, dim=0)[:, None]), dim=1)
                    priority = prior.numpy()
                    extra_time = full_ready + salience_time
                maximum = max(round(len(x) * (1 - budget)) for name, budget, weight in configs
                              if (name, weight) == key)
                if merge:
                    order, distances = select_centers(features, maximum, priority)
                else:
                    distances = torch.cdist(unit, unit)
                    if method == "random_node":
                        digest = hashlib.sha256(f"{seed}:{row['fact_id']}".encode()).digest()
                        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(
                            int.from_bytes(digest[:8], "little")))
                    elif method == "layer_importance":
                        layers = sorted(range(depth), key=lambda layer: -importance[layer])
                        order = torch.tensor([view * depth + layer for layer in layers for view in range(views)])
                    else:
                        view_order, _ = select_centers(unit.reshape(views, -1), views)
                        order = (view_order[:, None] * depth + torch.arange(depth)).flatten()
                plans[key] = order, distances, merge, perf_counter() - started + signal_time + extra_time
            order, distances, merge, plan_time = plans[key]
            started = perf_counter()
            compressed, assignment = compact_graph(x, edge, anchors, order[:count], distances, merge)
            construction_ms = (plan_time + perf_counter() - started) * 1000
            compact_x, compact_edge, compact_anchors = compressed
            signal = (x - compact_x[assignment]).square().sum() / x.square().sum().clamp_min(1e-12)
            spectral = spectral_error(compact_edge, assignment, basis, eigenvalues, count)
            result = dict(signal_relative_sse=signal.item(), spectral_relative_energy_mse=spectral)
            if profile:
                (compact_input, compact_pool, logits), cost = measured_probe(
                    model, compressed, final, device, profile)
                result["kl"] = F.kl_div(logits.log_softmax(-1), full_probs,
                                        reduction="batchmean").item()
                correction = (full_input - compact_input).norm(dim=1)
                logit_error = (logits - full_logits).abs().amax(dim=1)
                logit_bound = head_row_norm * correction
                pooled_bound = head_row_norm * output_norm * (full_pool - compact_pool).norm(dim=1)
                agreement = logits.argmax(1) == full_logits.argmax(1)
                result.update(nodes=count, edges=len(compact_edge), graph_build_ms=build_ms,
                              compression_ms=construction_ms, **cost,
                              graph_bytes=sum(map(tensor_bytes, (compact_x, compact_edge, compact_anchors))),
                              final_bytes=tensor_bytes(saved["states"][:, -1]),
                              top1_agreement=agreement.float().mean().item(),
                              correction_l2=correction.mean().item(), logit_error_linf=logit_error.mean().item(),
                              logit_bound_linf=logit_bound.mean().item(),
                              pooled_logit_bound_linf=pooled_bound.mean().item(),
                              top1_margin_fraction=(2 * logit_error < margin).float().mean().item(),
                              top1_certified_fraction=(2 * logit_bound < margin).float().mean().item(),
                              top1_pooled_certified_fraction=(2 * pooled_bound < margin).float().mean().item(),
                              observed_certificate_violations=((2 * logit_bound < margin) & ~agreement).sum().item())
                predictions[configuration_name(method, reduction, mix)].extend(records(row, logits))
            else:
                batch = prepare_batch(compressed, device)
                validation_inputs.append(model.vocabulary_input(batch, final, shared_graph=True))
                validation_results.append(result)
            metrics[configuration_name(method, reduction, mix)].append(result)
        if validation_inputs:
            entropy_term = (full_probs * full_logits.log_softmax(-1)).sum(-1)
            for start in range(0, len(validation_inputs), 8):
                states = torch.stack(validation_inputs[start:start + 8])
                logits = F.linear(states, model.unembedding, model.bias)
                kl = logits.logsumexp(-1) - (logits * full_probs).sum(-1) + entropy_term
                for offset, value in enumerate(kl.mean(-1).clamp_min(0).tolist()):
                    validation_results[start + offset]["kl"] = value
        if index % 100 == 0:
            print(f"{'Test' if profile else 'Validation'}: {index}/{len(dataset)} facts", flush=True)
    return {key: mean_metrics(rows) for key, rows in metrics.items()}, predictions


def run(cache, checkpoint, out, device, seed, calibration_facts, weights, ablations=False):
    device = torch.device(device)
    source = json.loads((cache / "source.json").read_text(encoding="utf-8"))
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if ("source" in saved and saved["source"] != source) or any(
            hashlib.sha256((cache / f"{split}.jsonl").read_bytes()).hexdigest() != digest
            for split, digest in saved.get("fit_manifests", {}).items()):
        raise ValueError("The checkpoint's model provenance or training/validation split differs from the cache.")
    options = saved["options"]
    if (options["model"] != "rgcn" or options["mode"] != "multi" or not options.get("residual_limit")
            or options.get("layers", "all") != "all" or options.get("random_edges")
            or not options.get("cross_view", True) or not options.get("inter_layer", True)
            or options.get("similarity")):
        raise ValueError("Compact requires the full all-layer residual R-GCN with canonical relation edges.")
    validation = Activations(cache, "val")
    if not len(validation) or min(validation[0][1]["states"].shape[:2]) < 2:
        raise ValueError("Compact needs nonempty validation and at least two views and layers.")
    head = torch.load(cache / "head.pt", weights_only=True)
    model = Probe(saved["size"], head, options, saved["pca"]).to(device).eval()
    model.load_state_dict(saved["state"])
    print("Calibrating layer importance on validation", flush=True)
    importance = layer_scores(model, validation, options, device, calibration_facts, seed)
    configs = [(method, reduction, mix) for method in METHODS for reduction in REDUCTIONS
               for mix in ((.1, 1.) if method in {"signal_aware", "prediction_aware"} else (0.,))]
    print(f"Scoring {len(configs)} configurations on {len(validation)} validation facts", flush=True)
    validation_metrics, _ = evaluate_compact(model, validation, options, configs, importance, device, seed, False)
    for metrics in validation_metrics.values():
        metrics["objective"] = sum(weight * metrics[name] for weight, name in
                                   zip(weights, ("signal_relative_sse", "spectral_relative_energy_mse", "kl"),
                                       strict=True))
    selected = defaultdict(list)
    for config in configs:
        selected[config[:2]].append(config)
    locked = [min(group, key=lambda config: validation_metrics[configuration_name(*config)]["objective"])
              for group in selected.values()]
    test = Activations(cache, "test")
    if not len(test):
        raise ValueError("Compact needs a nonempty test split.")
    print(f"Evaluating {len(locked)} locked configurations on {len(test)} test facts", flush=True)
    test_metrics, predictions = evaluate_compact(model, test, options, locked, importance, device, seed, True)
    reference = [row["prediction"] for row in predictions["full"]]
    for key, rows in predictions.items():
        test_metrics[key].update(accuracy(rows))
        if key != "full":
            test_metrics[key].update(paired_accuracy(rows, reference, np.random.default_rng(seed)))
    ablation_metrics = None
    if ablations:
        print(f"Running fixed-checkpoint graph ablations on {len(test)} test facts", flush=True)
        ablation_metrics = evaluate_ablations(model, test, options, device, reference, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(source=source, checkpoint=str(checkpoint), seed=seed, device=str(device),
                  hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                  torch_version=str(torch.__version__), options=options,
                  checkpoint_fit_manifests=saved.get("fit_manifests"),
                  facts={split: len(Activations(cache, split)) for split in ("train", "val", "test")},
                  weights=weights, validation_layer_calibration_facts=min(calibration_facts, len(validation)),
                  validation_layer_importance=importance, validation=validation_metrics,
                  selected=[configuration_name(*config) for config in locked],
                  frozen_accuracy=float(np.mean([row["base"] == row["gold"] for row in predictions["full"]])),
                  ablations=ablation_metrics, test=test_metrics)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
