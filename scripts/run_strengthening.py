"""Reproduce the seven follow-up experiments against local Llama and SmolLM2 caches."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from trace_k.causal import interventions, summarize_interventions
from trace_k.compact import evaluate_ablations, full_graph, prepare_batch
from trace_k.data import audit_partitions, partitions, read_jsonl, synthetic, write_jsonl
from trace_k.graph import Activations
from trace_k.llm import extract, load_model
from trace_k.probe import Probe
from trace_k.synthetic_adapter import adapt
from trace_k.train import (behavior_report, paired_accuracy, prediction_rows,
                           summarize, train)

LIMITS = (0, .01, .05, .10, .15, .25, .50)
ROOT = Path("runs/strengthening")
CACHE = Path("data/processed/llama-compact-full")
FULL = Path("runs/compact-full-controls")
OPTIONS = dict(model="rgcn", mode="multi", width=128, seed=42,
               view_dropout=.15, residual_limit=.15)


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def fit(cache, path, options, device, epochs=1):
    if (path / "metrics.json").is_file() and (path / "predictions.jsonl").is_file():
        return
    print(f"Training {path}", flush=True)
    train(cache, path, OPTIONS | options, device, epochs, 8, .001)


def metrics(rows):
    return summarize(rows) | paired_accuracy(rows, (r["base"] for r in rows),
                                              np.random.default_rng(42))


def align(rows, reference):
    keys = ("fact_id", "target", "prompt", "gold", "base")
    if len(rows) != len(reference) or any(
            any(a[k] != b[k] for k in keys) for a, b in zip(rows, reference, strict=True)):
        raise ValueError("Predictions refer to different evaluation prompts.")


def load_probe(cache, checkpoint, device):
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if saved["source"] != json.loads((cache / "source.json").read_text()) or any(
            hashlib.sha256((cache / f"{split}.jsonl").read_bytes()).hexdigest() != digest
            for split, digest in saved["fit_manifests"].items()):
        raise ValueError("The checkpoint was fitted on a different cache.")
    head = torch.load(cache / "head.pt", weights_only=True)
    model = Probe(saved["size"], head, saved["options"], saved["pca"]).to(device).eval()
    model.load_state_dict(saved["state"])
    return model, saved["options"]


@torch.no_grad()
def readouts(cache, checkpoint, device):
    """Sweep the inference cap without refitting and evaluate the mean frozen logits."""
    test = Activations(cache, "test")
    model, options = load_probe(cache, checkpoint, device)
    rows = {name: [] for name in (*LIMITS, "mean_logit")}
    for index, item in enumerate(test, 1):
        graph_data = prepare_batch(full_graph(item, options), device)
        final = item[1]["states"][:, -1].to(device, dtype=torch.float32)
        base = model.normalize(final)
        _, pooled = model.vocabulary_input(graph_data, final, return_pool=True, shared_graph=True)
        correction = model.output(pooled)
        ratio = base.norm(dim=-1, keepdim=True) / correction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        vectors = torch.stack([base + correction * (limit * ratio).clamp(max=1)
                               for limit in LIMITS] + [base.mean(dim=0).expand_as(base)])
        scores = F.linear(vectors, model.unembedding, model.bias).cpu()
        examples = [(item, i) for i in range(len(final))]
        for name, logits in zip(rows, scores, strict=True):
            rows[name].extend(prediction_rows(examples, logits))
        if index % 100 == 0:
            print(f"Residual sweep: {index}/{len(test)} facts", flush=True)
    return rows


def read_predictions(path):
    return read_jsonl(path / "predictions.jsonl")


def evaluate_sweep(device):
    rows = readouts(CACHE, FULL / "residual_rgcn/probe.pt", device)
    save(ROOT / "residual-strength.json", {str(cap): metrics(rows[cap]) for cap in LIMITS})
    write_jsonl(ROOT / "mean-logit.jsonl", rows["mean_logit"])


def evaluate_fair(device):
    for name, opts in (("residual_mlp", {"model": "pooled_mlp", "match_rgcn": True}),
                       ("residual_single", {"mode": "single"})):
        fit(CACHE, ROOT / name, opts, device)
    names = ("residual_rgcn", "residual_deepsets", "residual_transformer", "residual_mlp",
             "residual_single")
    paths = {name: (FULL if name in names[:3] else ROOT) / name for name in names}
    rows = {name: read_predictions(path) for name, path in paths.items()}
    rows["mean_logit"] = read_jsonl(ROOT / "mean-logit.jsonl")
    reference = rows["residual_rgcn"]
    facts = {r["fact_id"] for r in read_jsonl(CACHE / "test.jsonl")
             if r["gold"] not in r["base"] + r["heldout"]}
    fair = {}
    subset = {}
    for name, records in rows.items():
        align(records, reference)
        parameters = json.loads((paths[name] / "metrics.json").read_text())["parameters"] if name in paths else 0
        fair[name] = metrics(records) | {"parameters": parameters}
        fair[name]["versus_rgcn"] = paired_accuracy(
            records, (r["prediction"] for r in reference), np.random.default_rng(42))
        if name in ("residual_rgcn", "residual_single"):
            subset[name] = metrics([r for r in records if r["fact_id"] in facts])
    multi, single = ([r for r in rows[name] if r["fact_id"] in facts]
                     for name in ("residual_rgcn", "residual_single"))
    subset["multi_minus_single"] = paired_accuracy(
        multi, (r["prediction"] for r in single), np.random.default_rng(42))
    save(ROOT / "fair-baselines.json", fair)
    save(ROOT / "all-paraphrases-wrong.json", subset)


def evaluate_graph(device):
    model, options = load_probe(CACHE, FULL / "residual_rgcn/probe.pt", device)
    test = Activations(CACHE, "test")
    rows = read_predictions(FULL / "residual_rgcn")
    expected = [(row["fact_id"], target, prompt, row["gold"], row["base"][target])
                for row in test.rows for target, prompt in enumerate(row["paraphrases"])]
    fields = ("fact_id", "target", "prompt", "gold", "base")
    if expected != [tuple(r[field] for field in fields) for r in rows]:
        raise ValueError("Full checkpoint predictions do not match the test cache.")
    variants = evaluate_ablations(model, test, options, device,
                                  [r["prediction"] for r in rows], 42, edge_only=True)
    save(ROOT / "graph.json", dict(full=metrics(rows), variants=variants))


def evaluate_causal(device):
    source = json.loads((CACHE / "source.json").read_text())
    model, tokenizer = load_model(source["model_id"], device, source["dtype"])
    compact = json.loads(Path("reports/compact/full.json").read_text())
    # Independent validation calibration ranks probe layers by deletion KL.
    layer = int(max(compact["validation_layer_importance"],
                    key=compact["validation_layer_importance"].get))
    rng = random.Random(42)
    samples = {}
    for split, count in (("val", 64), ("test", 128)):
        facts = read_jsonl(CACHE / f"{split}.jsonl")
        rng.shuffle(facts)
        samples[split] = [dict(f, paraphrases=[rng.choice(f["paraphrases"])]) for f in facts[:count]]
    validation = []
    for count in (8, 16, 32):
        rows = list(interventions(model, tokenizer, samples["val"], layer, count,
                                  (1., 1.25, 1.5, 2.), 1, 42, True, False))
        validation.extend(dict(row, neuron_count=count) for row in summarize_interventions(rows))
    selected = max((row for row in validation if row["gain"] > 1),
                   key=lambda row: row["mean_delta_margin"])
    print(f"Causal validation selected {selected}", flush=True)
    rows = list(interventions(model, tokenizer, samples["test"], layer,
                              selected["neuron_count"], (1., selected["gain"]), 1, 42, True))
    write_jsonl(ROOT / "causal.jsonl", rows)
    save(ROOT / "causal.json", dict(layer=layer, validation_facts=64, test_facts=128,
                                  selected=selected, validation=validation,
                                  test=summarize_interventions(rows)))


def behavior_by_exposure(cache):
    result = {}
    for split in ("val", "test"):
        rows = read_jsonl(cache / f"{split}.jsonl")
        result[split] = {key: {str(label): behavior_report(
            [r for r in rows if r["exposed"] == label], scope) for label in (True, False)}
            for key, scope in (("trained_templates", "graph"), ("untrained_template", "heldout"))}
    return result


def exposure_metrics(rows):
    by_fact = {}
    for row in rows:
        label, probabilities = by_fact.setdefault(row["fact_id"], (row["exposed"], []))
        if label != row["exposed"]:
            raise ValueError("Exposure labels differ across views of a fact.")
        probabilities.append(row["gold_probability"])
    labels, probabilities = zip(*by_fact.values(), strict=True)
    return dict(exposure_fact_auroc=roc_auc_score(labels, [np.mean(p) for p in probabilities]),
                exposed_facts=sum(labels), unexposed_facts=len(labels) - sum(labels),
                exposed_accuracy=summarize([r for r in rows if r["exposed"]])["accuracy"],
                unexposed_accuracy=summarize([r for r in rows if not r["exposed"]])["accuracy"])


def evaluate_synthetic(device):
    root = ROOT / "synthetic"
    model_id = "data/raw/smollm2-135m"
    dtype = "bfloat16" if device.startswith("cuda") else "float32"
    facts = list(synthetic(["Berlin", "Paris", "Rome", "London"], 240, 42))
    # Three exposed prompt forms and one never shown to the adapter.
    for row in facts:
        row["behavior_templates"] = row["templates"][3:]
        row["behavior_prompts"] = row["paraphrases"][3:]
        row["templates"] = row["templates"][:3]
        row["paraphrases"] = row["paraphrases"][:3]
    write_jsonl(root / "facts.jsonl", facts)
    split = partitions(facts, "fact", 42)
    adapter = root / "adapter"
    if not (adapter / "adapter_config.json").exists():
        torch.manual_seed(42)
        adapt(root / "facts.jsonl", adapter, (model_id, device, dtype, None),
              epochs=35, batch_size=16, training_views=3, rank=32)
    behavior = {}
    for label, adapter_path in (("base", None), ("adapted", adapter)):
        cache = root / label
        if not (cache / "source.json").exists():
            model, tokenizer = load_model(model_id, device, dtype, adapter=adapter_path)
            timing = extract(model, tokenizer, split, cache, 8)
            save(cache / "source.json", dict(model_id=model_id, dtype=dtype,
                                            adapter=str(adapter_path) if adapter_path else None,
                                            extraction=timing))
            del model
        behavior[label] = behavior_by_exposure(cache)
        test = read_jsonl(cache / "test.jsonl")
        frozen = [dict(score, fact_id=fact["fact_id"], exposed=fact["exposed"],
                       gold=fact["gold"], prediction=token, base=token)
                  for fact in test for score, token in zip(
                      fact["behavior"][:len(fact["base"])], fact["base"], strict=True)]
        behavior[label]["frozen_graph_exposure"] = exposure_metrics(frozen)
    before = behavior["base"]["val"]["trained_templates"]["True"]["accuracy"]
    after = behavior["adapted"]["val"]["trained_templates"]["True"]["accuracy"]
    passed = after >= .5 and after - before >= .25
    report = dict(facts=len(facts), counts={name: len(rows) for name, rows in split.items()},
                  behavior=behavior, passed_learning_gate=passed,
                  learning_gate=dict(minimum_accuracy=.5, minimum_gain=.25), probes={})
    save(root / "report.json", report)
    if passed:
        for label in ("base", "adapted"):
            path = root / f"{label}-probe"
            fit(root / label, path, {}, device, epochs=3)
            rows = read_predictions(path)
            report["probes"][label] = metrics(rows) | exposure_metrics(rows)
        save(root / "report.json", report)


def evaluate_disjoint(device):
    report = {}
    for mode in ("subject", "template"):
        cache = Path(f"data/processed/strength-{mode}")
        path = ROOT / mode / "residual_rgcn"
        split = {name: read_jsonl(cache / f"{name}.jsonl") for name in ("train", "val", "test")}
        audit_partitions(split, mode)
        fit(cache, path, {}, device, epochs=3)
        result = json.loads((path / "metrics.json").read_text())
        report[mode] = dict(source=result["source"], paired=result["paired_test"],
                            partition_audit_passed=True,
                            counts={name: len(rows) for name, rows in split.items()})
    save(ROOT / "disjoint.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=("sweep", "fair", "causal", "graph", "synthetic", "disjoint"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    sections = dict(sweep=evaluate_sweep, fair=evaluate_fair, causal=evaluate_causal, graph=evaluate_graph,
                    synthetic=evaluate_synthetic, disjoint=evaluate_disjoint)
    sections[args.section](args.device)


if __name__ == "__main__":
    main()
