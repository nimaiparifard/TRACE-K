"""Probe training, checkpoint selection, and held-out evaluation."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from .data import read_jsonl, write_jsonl
from .graph import Activations, collate, graph
from .probe import Probe


def batches(dataset, size, shuffle=False):
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), size):
        yield [dataset[index] for index in indices[start:start + size]]


def samples(items, options, device, training=False, excluded_layer=None):
    examples = [(item, target) for item in items
                for target in range(len(item[0]["paraphrases"]))]
    if options["model"] in {"linear", "mlp", "deepsets", "transformer", "tuned"}:
        options = options | dict(inter_layer=False, cross_view=False, similarity=False, random_edges=False)
    return examples, collate([graph(item, target, options, training, excluded_layer)
                             for item, target in examples], device)


def final_states(examples, device):
    return torch.stack([item[1]["states"][target, -1]
                        for item, target in examples]).to(device, dtype=torch.float32)


def summarize(rows):
    def average(values):
        return sum(values) / len(values) if values else None

    correct = [row["prediction"] == row["gold"] for row in rows]
    failed = [hit for hit, row in zip(correct, rows) if row["base"] != row["gold"]]
    expressed = [hit for hit, row in zip(correct, rows) if row["base"] == row["gold"]]
    by_fact = {}
    for row in rows:
        by_fact.setdefault(row["fact_id"], []).append(row["prediction"])
    multi_view = [group for group in by_fact.values() if len(group) > 1]
    bins = [[(hit, row["confidence"]) for hit, row in zip(correct, rows)
             if min(9, int(row["confidence"] * 10)) == index] for index in range(10)]
    return dict(accuracy=average(correct), hits_at_1=average(correct),
                hits_at_5=average([row["rank"] <= 5 for row in rows]),
                mrr=average([1 / row["rank"] for row in rows]),
                pairwise_accuracy=average([score for row in rows for score in row["pairwise"]]),
                recovery_rate=average(failed), failed_prompts=len(failed),
                expressed_regression_rate=average([not hit for hit in expressed]),
                ece=sum(abs(sum(hit - prob for hit, prob in group)) for group in bins) / len(rows),
                consistency=average([len(set(group)) == 1 for group in multi_view]),
                multi_view_facts=len(multi_view),
                exposure_auroc=(roc_auc_score([row["exposed"] for row in rows],
                                              [row["gold_probability"] for row in rows])
                                if {row.get("exposed") for row in rows} == {True, False} else None),
                count=len(rows))


def behavior_report(facts, scope="all"):
    rows = []
    for fact in facts:
        measured = zip(fact["base"] + fact["heldout"], fact["behavior"], strict=True)
        for index, (prediction, measurement) in enumerate(measured):
            graph_view = index < len(fact["base"])
            if scope == "all" or (scope == "graph" and graph_view) or (scope == "heldout" and not graph_view):
                rows.append(dict(measurement, fact_id=fact["fact_id"], gold=fact["gold"],
                                 prediction=prediction, base=prediction,
                                 exposed=fact.get("exposed")))
    if not rows:
        return None
    metrics = summarize(rows)
    for key in ("recovery_rate", "expressed_regression_rate"):
        del metrics[key]
    return metrics


def intermediate_lens_report(validation, test):
    """Select an intermediate layer on validation data and report frozen test accuracy."""
    validation_hits = np.asarray([hits for fact in validation for hits in fact["lens_correct"]])
    if validation_hits.shape[1] < 2:
        return None
    validation_accuracy = validation_hits.mean(axis=0)
    layer = int(validation_accuracy[:-1].argmax())
    hits = [layers[layer] for fact in test for layers in fact["lens_correct"]]
    frozen_hits = [prediction == fact["gold"] for fact in test for prediction in fact["base"]]
    wrong = [hit for hit, frozen in zip(hits, frozen_hits, strict=True) if not frozen]
    correct = [hit for hit, frozen in zip(hits, frozen_hits, strict=True) if frozen]
    return dict(layer=layer, validation_accuracy=float(validation_accuracy[layer]),
                accuracy=sum(hits) / len(hits), count=len(hits), failed_prompts=len(wrong),
                recovery_rate=sum(wrong) / len(wrong) if wrong else None,
                expressed_regression_rate=1 - sum(correct) / len(correct) if correct else None)


def paired_accuracy(rows, reference_tokens, rng):
    per_fact = defaultdict(lambda: [0, 0])
    for row, reference in zip(rows, reference_tokens, strict=True):
        totals = per_fact[row["fact_id"]]
        totals[0] += int(row["prediction"] == row["gold"]) - int(reference == row["gold"])
        totals[1] += 1
    changes, counts = np.array(list(per_fact.values())).T
    interval = None
    if len(counts) > 1:
        indices = rng.integers(len(counts), size=(2000, len(counts)))
        estimates = changes[indices].sum(axis=1) / counts[indices].sum(axis=1)
        interval = np.quantile(estimates, [.025, .975]).tolist()
    return dict(prompts=len(rows), facts=len(counts),
                accuracy_delta=float(changes.sum() / counts.sum()), accuracy_delta_ci95=interval)


def compare_runs(reference, candidates, seed):
    reference_rows = read_jsonl(reference / "predictions.jsonl")
    fields = ("fact_id", "target", "prompt", "gold", "base")

    def signature(rows):
        return [tuple(row[field] for field in fields) for row in rows]

    reference_signature = signature(reference_rows)
    result = {}
    for path in candidates:
        rows = read_jsonl(path / "predictions.jsonl")
        if signature(rows) != reference_signature:
            raise ValueError(f"Test prompts and frozen predictions differ in {path}.")
        result[str(path)] = paired_accuracy(rows, (row["prediction"] for row in reference_rows),
                                            np.random.default_rng(seed))
    return dict(reference=str(reference), comparisons=result)


def paired_results(rows, training_tokens, seed):
    """Compare on identical prompts and resample entire facts for the interval."""
    groups = {"all": rows,
              "seen_answer": [row for row in rows if row["gold"] in training_tokens],
              "unseen_answer": [row for row in rows if row["gold"] not in training_tokens]}
    result = {}
    rng = np.random.default_rng(seed)
    for label, group in groups.items():
        if not group:
            result[label] = None
            continue
        result[label] = dict(frozen_accuracy=sum(row["base"] == row["gold"] for row in group) / len(group),
                             **paired_accuracy(group, (row["base"] for row in group), rng),
                             **summarize(group))
    return result


@torch.no_grad()
def evaluate(model, dataset, options, device, batch_size, excluded_layer=None):
    model.eval()
    rows = []
    for items in batches(dataset, batch_size):
        examples, batch = samples(items, options, device, excluded_layer=excluded_layer)
        final = final_states(examples, device) if model.residual_limit else None
        logits = model(batch, final).cpu()
        for ((fact, _), target), scores in zip(examples, logits):
            gold = fact["gold"]
            prediction = scores.argmax().item()
            probability = scores.softmax(-1)
            base = fact["base"][target]
            if base == gold:
                evidence = "Expressed Knowledge"
            elif gold in fact["base"] + fact["heldout"]:
                evidence = "Accessible but Inconsistent"
            elif prediction == gold:
                evidence = "Internally Decodable"
            else:
                evidence = "No Evidence Found"
            tied_before = (scores[:gold] == scores[gold]).sum().item()
            rows.append(dict(fact_id=fact["fact_id"], target=target,
                             prompt=fact["paraphrases"][target], answer=fact["object"],
                             base=base, base_text=fact["base_text"][target], gold=gold,
                             prediction=prediction, confidence=probability[prediction].item(),
                             gold_probability=probability[gold].item(), evidence=evidence,
                             exposed=fact.get("exposed"), lens_correct=fact["lens_correct"][target],
                             rank=1 + (scores > scores[gold]).sum().item() + tied_before,
                             pairwise=(scores[gold] > scores[fact["negatives"]]).tolist()))
    return summarize(rows), rows


def train(cache, out, options, device, epochs, batch_size, learning_rate):
    torch.manual_seed(options["seed"])
    random.seed(options["seed"])
    cache, out = Path(cache), Path(out)
    source = json.loads((cache / "source.json").read_text(encoding="utf-8"))
    fit_manifests = {split: hashlib.sha256((cache / f"{split}.jsonl").read_bytes()).hexdigest()
                     for split in ("train", "val")}
    train_set, val_set, test_set = [Activations(cache, name) for name in ("train", "val", "test")]
    if options["model"] == "tuned":
        depth = train_set[0][1]["states"].size(1)
        options = options | {"layers": (depth - 1) // 2, "mode": "single",
                             "features": "residual", "pca": 0}
    feature_size = graph(train_set[0], 0, options)[0].shape[-1]
    head = torch.load(cache / "head.pt", weights_only=True)
    pca = None
    if options.get("pca", 0):
        subset = [graph(train_set[index], 0, options)[0]
                  for index in range(min(len(train_set), 128))]
        pool = torch.cat(subset)[:8192]
        mean = pool.mean(0)
        _, _, vectors = torch.pca_lowrank(pool - mean, q=min(options["pca"], *pool.shape),
                                           center=False)
        pca = mean, vectors.T.contiguous()
    model = Probe(feature_size, head, options, pca).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    labels = [row["gold"] for row in train_set.rows]
    training_tokens = set(labels)
    shuffled = labels.copy()
    if options.get("shuffle_labels"):
        random.shuffle(shuffled)
    by_fact = {row["fact_id"]: gold for row, gold in zip(train_set.rows, shuffled, strict=True)}
    out.mkdir(parents=True, exist_ok=False)
    history = []
    best = -1
    for epoch in range(epochs):
        model.train()
        total, count = 0., 0
        for items in batches(train_set, batch_size, shuffle=True):
            examples, batch = samples(items, options, device, training=True)
            gold = torch.tensor([by_fact[item[0]["fact_id"]] for item, _ in examples], device=device)
            optimizer.zero_grad(set_to_none=True)
            final = final_states(examples, device) if model.residual_limit or model.kind == "tuned" else None
            logits = model(batch, final)
            if model.kind == "tuned":
                teacher = model.lens(final).softmax(-1)
                loss = F.kl_div(logits.log_softmax(-1), teacher, reduction="batchmean")
            else:
                loss = F.cross_entropy(logits, gold)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(gold)
            count += len(gold)
        metrics, _ = evaluate(model, val_set, options, device, batch_size)
        history.append(dict(epoch=epoch + 1, train_loss=total / count, validation=metrics))
        print(f"Epoch {epoch + 1}/{epochs} | loss {total / count:.4f} | "
              f"validation MRR {metrics['mrr']:.4f}", flush=True)
        if metrics["mrr"] > best:
            best = metrics["mrr"]
            torch.save(dict(state=model.state_dict(), options=options, pca=pca,
                            size=feature_size, source=source, fit_manifests=fit_manifests),
                       out / "probe.pt")
    model.load_state_dict(torch.load(out / "probe.pt", weights_only=True)["state"])
    metrics, predictions = evaluate(model, test_set, options, device, batch_size)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cache / "tokenizer")
    texts = tokenizer.batch_decode([[row["prediction"]] for row in predictions])
    for row, prediction_text in zip(predictions, texts, strict=True):
        row["prediction_text"] = prediction_text
        row["answer_seen_in_train"] = row["gold"] in training_tokens
    lens_hits = torch.tensor([row["lens_correct"] for row in predictions], dtype=torch.float32)
    report = dict(options=options, source=source, cache=str(cache.resolve()),
                  parameters=sum(parameter.numel() for parameter in model.parameters()),
                  history=history, test=metrics, frozen_llm=behavior_report(test_set.rows, "graph"),
                  paired_test=paired_results(predictions, training_tokens, options["seed"]),
                  frozen_llm_heldout=behavior_report(test_set.rows, "heldout"),
                  logit_lens_accuracy_by_layer=lens_hits.mean(0).tolist())
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_jsonl(out / "predictions.jsonl", predictions)


@torch.no_grad()
def layer_importance(cache, checkpoint, device, batch_size):
    cache = Path(cache)
    saved = torch.load(checkpoint, weights_only=True)
    head = torch.load(cache / "head.pt", weights_only=True)
    options = saved["options"]
    model = Probe(saved["size"], head, options, saved["pca"]).to(device)
    model.load_state_dict(saved["state"])
    dataset = Activations(cache, "val")
    _, baseline = evaluate(model, dataset, options, device, batch_size)
    depth = dataset[0][1]["states"].size(1)
    if depth < 2 or options.get("layers", "all") != "all":
        raise ValueError("Layer importance requires an all-layer probe with at least two layers.")
    for layer in range(depth):
        _, modified = evaluate(model, dataset, options, device, batch_size, layer)
        differences = [before["gold_probability"] - after["gold_probability"]
                       for before, after in zip(baseline, modified, strict=True)]
        yield dict(layer=layer, validation_probability_drop=sum(differences) / len(differences))
