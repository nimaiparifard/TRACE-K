"""Exercise real decoder hooks and the full local experiment path."""

import json
import sys
from collections import Counter

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from trace_k.cli import main
from trace_k.compact import full_graph, prepare_batch
from trace_k.data import partitions, read_jsonl, synthetic
from trace_k.graph import Activations, collate, graph
from trace_k.probe import Probe


@pytest.fixture
def run(monkeypatch):
    def command(*args):
        monkeypatch.setattr(sys, "argv", ["trace-k", *map(str, args)])
        main()
    return command


@pytest.fixture
def model_path(tmp_path):
    torch.set_num_threads(1)
    model_path = tmp_path / "tiny"
    vocab = {word: index for index, word in enumerate(
        ["[UNK]", "[PAD]", "[EOS]", "Berlin", "Paris", "Rome", "London", "is", "."])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                       pad_token="[PAD]", eos_token="[EOS]",
                                       model_input_names=["input_ids", "attention_mask"])
    tokenizer.save_pretrained(model_path)
    LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=24,
                                num_hidden_layers=3, num_attention_heads=2,
                                num_key_value_heads=2, pad_token_id=1,
                                eos_token_id=2)).save_pretrained(model_path)
    return model_path


def test_pipeline(tmp_path, run, model_path):
    facts, splits, cache = (tmp_path / name for name in ("facts.jsonl", "splits", "cache"))
    runs, causal, importance = (tmp_path / name for name in ("runs", "causal.jsonl", "layers.jsonl"))
    run("synthetic", facts, "--count", 12)
    run("split", facts, splits)
    run("extract", splits, cache, "--model-id", model_path, "--device", "cpu")
    extraction = json.loads((cache / "source.json").read_text())["extraction"]
    assert extraction["prompts"] > 0
    assert extraction["cuda_peak_allocated_bytes"] is None
    row, saved = Activations(cache, "train")[0]
    expected = torch.tensor([item["gold_probability"] for item in row["behavior"]])
    torch.testing.assert_close(saved["lens_gold_probability"][:, -1], expected)
    behavior = tmp_path / "behavior.json"
    run("behavior", cache, behavior)
    assert json.loads(behavior.read_text())["test"]["count"] > 0
    run("suite", cache, runs, "--device", "cpu", "--epochs", 1)
    run("importance", cache, runs / "rgcn" / "probe.pt", importance, "--device", "cpu")
    assert len(read_jsonl(importance)) == 3
    assert {path.parent.name for path in runs.rglob("metrics.json")} >= {"rgcn", "single", "pca"}
    run("causal", cache, causal, "--device", "cpu", "--importance", importance, "--limit", 2,
        "--neurons", 2, "--steps", 2, "--gains", 1, 2)
    interventions = read_jsonl(causal)
    assert {"attributed", "random", "activation_matched", "other_layer"} <= {
        row["control"] for row in interventions}
    assert all(row["delta_margin"] == pytest.approx(0, abs=1e-6)
               for row in interventions if row["gain"] == 1)
    metrics = json.loads((runs / "rgcn" / "metrics.json").read_text())
    assert metrics["test"]["count"] > 0
    assert metrics["paired_test"]["all"]["prompts"] == metrics["test"]["count"]
    test_facts = read_jsonl(cache / "test.jsonl")
    all_wrong = [row for row in test_facts if row["gold"] not in row["base"] + row["heldout"]]
    subset = metrics["paired_test"]["all_paraphrases_wrong"]
    if all_wrong:
        assert subset["facts"] == len(all_wrong)
        assert subset["prompts"] == sum(len(row["paraphrases"]) for row in all_wrong)
        assert subset["frozen_accuracy"] == 0
    else:
        assert subset is None
    assert (runs / "residual_mlp" / "metrics.json").is_file()
    comparison = tmp_path / "comparison.json"
    run("compare", runs / "rgcn", comparison, runs / "residual_rgcn")
    paired = json.loads(comparison.read_text())["comparisons"][str(runs / "residual_rgcn")]
    assert paired["prompts"] == metrics["test"]["count"]
    checkpoint = torch.load(runs / "residual_rgcn" / "probe.pt", weights_only=True)
    model = Probe(checkpoint["size"], torch.load(cache / "head.pt", weights_only=True),
                  checkpoint["options"]).eval()
    model.load_state_dict(checkpoint["state"])
    item = Activations(cache, "test")[0]
    final = item[1]["states"][:, -1].float()
    repeated = collate([graph(item, target, checkpoint["options"]) for target in range(len(final))], "cpu")
    shared = prepare_batch(full_graph(item, checkpoint["options"]), "cpu")
    torch.testing.assert_close(model.vocabulary_input(repeated, final),
                               model.vocabulary_input(shared, final, shared_graph=True))
    compact_path = tmp_path / "compact.json"
    run("compact", cache, runs / "residual_rgcn" / "probe.pt", compact_path,
        "--device", "cpu", "--calibration-facts", 1, "--ablations")
    compact = json.loads(compact_path.read_text())
    assert len(compact["selected"]) == 28
    assert "leave_out_layer_2" in compact["ablations"]
    assert compact["test"]["full"]["accuracy"] == json.loads(
        (runs / "residual_rgcn" / "metrics.json").read_text())["test"]["accuracy"]
    for name in compact["selected"]:
        assert compact["test"][name]["nodes"] < compact["test"]["full"]["nodes"]
        assert compact["test"][name]["observed_certificate_violations"] == 0
    assert all(isinstance(row["prediction_text"], str)
               for row in read_jsonl(runs / "rgcn" / "predictions.jsonl"))


def test_adapter(tmp_path, run, model_path):
    pytest.importorskip("peft")
    facts, splits = tmp_path / "facts.jsonl", tmp_path / "splits"
    run("synthetic", facts, "--count", 12)
    run("split", facts, splits)
    adapter = tmp_path / "adapter"
    run("adapt", facts, adapter, "--model-id", model_path, "--device", "cpu", "--epochs", 1)
    adapted = tmp_path / "adapted"
    run("extract", splits, adapted, "--model-id", model_path, "--device", "cpu", "--adapter", adapter)
    assert json.loads((adapted / "source.json").read_text())["adapter"] == str(adapter)


def test_group_overlap_and_graph_ablation():
    rows = list(synthetic(["A", "B", "C"], 30, 2))
    split = partitions(rows, "object", 42)
    label_sets = [{row["object"] for row in group} for group in split.values()]
    assert not any(first & second for index, first in enumerate(label_sets)
                   for second in label_sets[index + 1:])
    item = rows[0], {"states": torch.ones(4, 3, 4)}
    x, edges, anchor = graph(item, 2, {"mode": "multi", "cross_view": False})
    assert x.shape == (12, 4)
    assert anchor == 8
    assert set(edges[:, 2].tolist()) == {0}
    _, rewired, _ = graph(item, 2, {"mode": "multi", "cross_view": False, "random_edges": True})
    for column in (0, 1):
        assert Counter(edges[:, column].tolist()) == Counter(rewired[:, column].tolist())
    assert edges.tolist() != rewired.tolist()


def test_residual_probe_initialization_and_bound():
    head = dict(weight=torch.eye(4), bias=None, norm_weight=torch.ones(4), norm_epsilon=1e-6)
    options = dict(model="linear", width=2, residual_limit=.15)
    probe = Probe(4, head, options)
    final = torch.tensor([[1., -3., 2., 1.]])
    batch = (final, None, None, None, torch.tensor([0]))
    reference = probe.lens(final)
    torch.testing.assert_close(probe(batch, final), reference)
    with torch.no_grad():
        probe.output.bias.fill_(100)
    assert (probe(batch, final) - reference).norm() <= reference.norm() * .15001
