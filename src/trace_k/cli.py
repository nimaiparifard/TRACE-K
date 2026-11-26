"""Command-line entry points for TRACE-K experiments."""

import argparse
import json
import random
from pathlib import Path

import torch
import yaml

from .data import audit_partitions, pararel, partitions, read_jsonl, synthetic, write_jsonl


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Convert a ParaRel checkout to JSONL.")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("out", type=Path)
    prepare.add_argument("--views", type=positive, default=8)
    prepare.add_argument("--behavior-views", type=int, default=0)
    synth = commands.add_parser("synthetic", help="Generate balanced random factual associations.")
    synth.add_argument("out", type=Path)
    synth.add_argument("--count", type=positive, default=120)
    synth.add_argument("--objects", nargs="+", default=["Berlin", "Paris", "Rome", "London"])
    split = commands.add_parser("split", help="Create leakage-controlled JSONL partitions.")
    split.add_argument("facts", type=Path)
    split.add_argument("out", type=Path)
    split.add_argument("--mode", choices=["fact", "subject", "relation", "object", "template"], default="fact")
    split.add_argument("--limit", type=positive)
    extraction = commands.add_parser("extract", help="Cache frozen decoder representations.")
    extraction.add_argument("splits", type=Path)
    extraction.add_argument("out", type=Path)
    behavior = commands.add_parser("behavior", help="Report frozen LLM next-token behavior.")
    behavior.add_argument("cache", type=Path)
    behavior.add_argument("out", type=Path)
    training = commands.add_parser("train", help="Train and evaluate a single probe.")
    suite = commands.add_parser("suite", help="Run baselines and graph ablations.")
    suite.add_argument("--random-cache", type=Path, help="Matching cache from randomly initialized weights.")
    suite.add_argument("--variants", nargs="+", metavar="NAME",
                       help="Run only the named suite variants.")
    comparison = commands.add_parser("compare", help="Compare checkpoints on identical test prompts.")
    comparison.add_argument("reference", type=Path)
    comparison.add_argument("out", type=Path)
    comparison.add_argument("candidates", nargs="+", type=Path)
    for command in (training, suite):
        command.add_argument("cache", type=Path)
        command.add_argument("out", type=Path)
        command.add_argument("--config", type=Path)
        command.add_argument("--epochs", type=positive, default=5)
        command.add_argument("--learning-rate", type=float, default=.001)
    importance = commands.add_parser("importance", help="Score layer removal on validation data.")
    importance.add_argument("cache", type=Path)
    importance.add_argument("checkpoint", type=Path)
    importance.add_argument("out", type=Path)
    causal = commands.add_parser("causal", help="Evaluate neuron scaling and matched controls.")
    causal.add_argument("cache", type=Path)
    causal.add_argument("out", type=Path)
    selection = causal.add_mutually_exclusive_group(required=True)
    selection.add_argument("--layer", type=int)
    selection.add_argument("--importance", type=Path)
    causal.add_argument("--limit", type=positive, default=20)
    causal.add_argument("--neurons", type=positive, default=32)
    causal.add_argument("--steps", type=positive, default=1)
    causal.add_argument("--gains", nargs="+", type=float, default=[0., .5, 1., 1.5, 2.])
    adapt = commands.add_parser("adapt", help="Train a LoRA adapter on exposed synthetic facts.")
    adapt.add_argument("facts", type=Path)
    adapt.add_argument("out", type=Path)
    adapt.add_argument("--epochs", type=positive, default=3)
    for command in (extraction, adapt):
        command.add_argument("--model-id", default="meta-llama/Llama-3.2-1B")
        command.add_argument("--revision")
        command.add_argument("--dtype", choices=["float32", "bfloat16", "float16"])
    extraction.add_argument("--random-weights", action="store_true")
    extraction.add_argument("--adapter")
    for command in (extraction, training, suite, importance, causal, adapt):
        command.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        command.add_argument("--batch-size", type=positive, default=2)
    for command in (split, synth, extraction, training, suite, causal, adapt, comparison):
        command.add_argument("--seed", type=int, default=42)
    return root


def model_options(args):
    return (args.model_id, args.device,
            args.dtype or ("bfloat16" if args.device.startswith("cuda") else "float32"),
            args.revision)


def check_random_cache(reference, control):
    first, second = (json.loads((path / "source.json").read_text(encoding="utf-8"))
                     for path in (reference, control))
    fields = ("model_id", "resolved_revision", "revision", "dtype")
    if first["random_weights"] or not second["random_weights"] or any(
            first[field] != second[field] for field in fields):
        raise ValueError("The random-weight cache must match the pretrained model, revision and dtype.")
    for name in ("train", "val", "test"):
        signature = []
        for path in (reference, control):
            signature.append([(row["fact_id"], row["paraphrases"], row["gold"])
                              for row in read_jsonl(path / f"{name}.jsonl")])
        if signature[0] != signature[1]:
            raise ValueError(f"The random-weight cache differs in the {name} split.")


def main():
    args = parser().parse_args()
    command = args.command
    if hasattr(args, "seed"):
        torch.manual_seed(args.seed)
    if command == "prepare":
        rows = list(pararel(args.source, args.views, args.behavior_views))
        if not rows:
            raise ValueError("No usable ParaRel facts found for the requested view count.")
        write_jsonl(args.out, rows)
        print(f"Prepared {len(rows)} facts.")
    elif command == "synthetic":
        write_jsonl(args.out, synthetic(args.objects, args.count, args.seed))
    elif command == "split":
        rows = read_jsonl(args.facts)
        for name, facts in partitions(rows, args.mode, args.seed, args.limit).items():
            write_jsonl(args.out / f"{name}.jsonl", facts)
            print(f"{name}: {len(facts)} facts")
        (args.out / "split.json").write_text(json.dumps(dict(mode=args.mode, seed=args.seed,
                                                            limit=args.limit)),
                                             encoding="utf-8")
    elif command == "extract":
        from .llm import extract, load_model

        splits = {name: read_jsonl(args.splits / f"{name}.jsonl") for name in ("train", "val", "test")}
        split_file = args.splits / "split.json"
        partition = json.loads(split_file.read_text()) if split_file.exists() else {"mode": "fact"}
        audit_partitions(splits, partition["mode"])
        if args.random_weights and args.adapter:
            raise ValueError("Random-weight and adapted checkpoints are separate controls.")
        model, tokenizer = load_model(*model_options(args), args.random_weights, args.adapter)
        extract(model, tokenizer, splits, args.out, args.batch_size)
        metadata = dict(model_id=args.model_id, revision=args.revision,
                        resolved_revision=getattr(model.config, "_commit_hash", None),
                        dtype=model_options(args)[2], random_weights=args.random_weights,
                        adapter=args.adapter, seed=args.seed, split=partition)
        (args.out / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    elif command == "behavior":
        from .train import behavior_report, intermediate_lens_report

        facts = {name: read_jsonl(args.cache / f"{name}.jsonl") for name in ("train", "val", "test")}
        report = {name: behavior_report(rows) for name, rows in facts.items()}
        report["intermediate_logit_lens"] = intermediate_lens_report(facts["val"], facts["test"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    elif command in {"train", "suite"}:
        from .train import train

        options = dict(model="rgcn", mode="multi", width=128, seed=args.seed, view_dropout=.15)
        if args.config:
            configured = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            if not isinstance(configured, dict):
                raise ValueError("The probe configuration must be a YAML mapping.")
            options.update(configured)
        if options["mode"] not in {"single", "multi"}:
            raise ValueError("Probe mode must be single or multi.")
        if options["model"] not in {"rgcn", "gcn", "sage", "gat", "linear", "mlp",
                                   "deepsets", "transformer", "tuned"}:
            raise ValueError("Unknown probe model.")
        if not 0 <= options["view_dropout"] < 1 or options["width"] < 1:
            raise ValueError("Set a positive probe width and a view dropout in [0, 1).")
        if command == "train":
            variants = {"": {}}
        else:
            variants = {model: {"model": model} for model in
                        ("linear", "mlp", "deepsets", "transformer", "gcn", "sage", "gat", "rgcn")}
            variants.update({"tuned_lens": {"model": "tuned"},
                             "residual_rgcn": {"model": "rgcn", "residual_limit": .15},
                             "residual_linear": {"model": "linear", "mode": "single",
                                                 "layers": "last", "residual_limit": .15},
                             "matched_mlp": {"model": "mlp", "match_rgcn": True},
                             "single": {"mode": "single"}, "no_cross_view": {"cross_view": False},
                             "no_inter_layer": {"inter_layer": False}, "random_graph": {"random_edges": True},
                             "last_layer": {"layers": "last"}, "middle_layers": {"layers": "middle"},
                             "statistics": {"features": "stats"}, "residual_only": {"features": "residual"},
                             "no_dropout": {"view_dropout": 0}, "embedding": {"features": "embedding"},
                             "shuffled_labels": {"shuffle_labels": True}, "similarity": {"similarity": True},
                             "pca": {"pca": 128}})
            if args.random_cache:
                check_random_cache(args.cache, args.random_cache)
                variants["random_llm"] = {}
            if args.variants:
                missing = set(args.variants) - variants.keys()
                if missing:
                    raise ValueError(f"Unknown suite variants: {', '.join(sorted(missing))}")
                variants = {name: variants[name] for name in args.variants}
        for name, overrides in variants.items():
            print(f"Training {name or options['model']}", flush=True)
            cache = args.random_cache if name == "random_llm" else args.cache
            train(cache, args.out / name, options | overrides,
                  args.device, args.epochs, args.batch_size, args.learning_rate)
    elif command == "compare":
        from .train import compare_runs

        report = compare_runs(args.reference, args.candidates, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    elif command == "importance":
        from .train import layer_importance

        write_jsonl(args.out, layer_importance(args.cache, args.checkpoint, args.device, args.batch_size))
    elif command == "causal":
        from .causal import interventions, summarize_interventions
        from .llm import load_model

        source = json.loads((args.cache / "source.json").read_text(encoding="utf-8"))
        args.model_id = source["model_id"]
        args.revision = source["resolved_revision"] or source["revision"]
        args.dtype = source["dtype"]
        torch.manual_seed(source["seed"])
        model, tokenizer = load_model(*model_options(args), source["random_weights"], source["adapter"])
        layer = (args.layer if args.importance is None else
                 max(read_jsonl(args.importance), key=lambda row: row["validation_probability_drop"])["layer"])
        facts = read_jsonl(args.cache / "test.jsonl")
        random.Random(args.seed).shuffle(facts)
        rows = list(interventions(model, tokenizer, facts[:args.limit], layer,
                                  args.neurons, args.gains, args.steps, args.seed))
        write_jsonl(args.out, rows)
        args.out.with_suffix(".summary.json").write_text(
            json.dumps(summarize_interventions(rows), indent=2), encoding="utf-8")
    elif command == "adapt":
        from .synthetic_adapter import adapt

        adapt(args.facts, args.out, model_options(args), args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
