"""Frozen decoder activations and next-token measurements."""

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .data import write_jsonl


def load_model(model_id, device, dtype, revision=None, random_weights=False, adapter=None):
    options = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **options)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    precision = getattr(torch, dtype)
    if random_weights:
        config = AutoConfig.from_pretrained(model_id, **options)
        model = AutoModelForCausalLM.from_config(config).to(dtype=precision)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=precision, **options)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model = model.to(device).eval().requires_grad_(False)
    if not hasattr(model.model, "layers") or not hasattr(model.model, "norm"):
        raise ValueError("TRACE-K requires a Llama-style decoder with layers and final norm.")
    return model, tokenizer


def continuation_id(tokenizer, prompts, answer):
    suffix = " " + answer.strip()
    result = None
    for prompt in prompts:
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        full = tokenizer.encode(prompt + suffix, add_special_tokens=False)
        if not full or full[:-1] != prefix or full[-1] in tokenizer.all_special_ids:
            return None
        if result is not None and result != full[-1]:
            return None
        result = full[-1]
    return result


def inputs_for(tokenizer, prompts, device):
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    inputs.pop("token_type_ids", None)
    inputs["position_ids"] = (inputs.attention_mask.cumsum(dim=1) - 1).clamp_min(0)
    positions = torch.arange(inputs.input_ids.size(1), device=device)
    last = positions.masked_fill(inputs.attention_mask == 0, -1).amax(dim=1)
    return inputs, last


@torch.no_grad()
def representations(model, tokenizer, prompts, batch_size, gold, negatives):
    residuals, embeddings, predictions, lens_predictions, entropy = [], [], [], [], []
    gold_curves, base_curves = [], []
    measurements = []
    head = model.get_output_embeddings()
    for start in range(0, len(prompts), batch_size):
        inputs, last = inputs_for(tokenizer, prompts[start:start + batch_size], model.device)
        batch = torch.arange(len(last), device=model.device)
        layers = []

        def collect(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            layers.append(hidden[batch, last])

        handles = [layer.register_forward_hook(collect) for layer in model.model.layers]
        try:
            output = model(**inputs, use_cache=False, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        states = torch.stack(layers, dim=1)
        residuals.append(states.cpu().to(torch.bfloat16))
        embeddings.append(model.get_input_embeddings()(inputs.input_ids[batch, last]).cpu()
                          .to(torch.bfloat16))
        logits = output.logits[:, -1].float()
        tops = logits.argmax(-1)
        predictions.append(tops.cpu())
        probabilities = logits.softmax(-1)
        gold_scores = logits[:, gold:gold + 1]
        rank = 1 + (logits > gold_scores).sum(-1) + (logits[:, :gold] == gold_scores).sum(-1)
        for i in range(len(tops)):
            measurements.append(dict(rank=rank[i].item(), confidence=probabilities[i, tops[i]].item(),
                                     gold_probability=probabilities[i, gold].item(),
                                     pairwise=(gold_scores[i] > logits[i, negatives]).tolist()))
        lens = head(model.model.norm(states)).float().log_softmax(-1)
        lens_predictions.append(lens.argmax(-1).cpu())
        gold_curves.append(lens[..., gold].exp().cpu())
        base_curves.append(lens.gather(-1, tops[:, None, None].expand(-1, states.size(1), 1))
                           .squeeze(-1).exp().cpu())
        entropy.append(-(lens.exp() * lens).sum(-1).cpu())
    return (*[torch.cat(items) for items in
              (residuals, embeddings, predictions, lens_predictions, entropy, gold_curves, base_curves)],
            measurements)


def extract(model, tokenizer, split, directory, batch_size):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(directory / "tokenizer")
    head = model.get_output_embeddings()
    norm = model.model.norm
    torch.save({"weight": head.weight.detach().cpu(),
                "bias": None if head.bias is None else head.bias.detach().cpu(),
                "norm_weight": norm.weight.detach().cpu(),
                "norm_epsilon": norm.variance_epsilon}, directory / "head.pt")
    for name, facts in split.items():
        manifest = []
        for fact in facts:
            prompts = [prompt.rstrip() for prompt in fact["paraphrases"]]
            heldout = [prompt.rstrip() for prompt in fact.get("behavior_prompts", [])]
            gold = continuation_id(tokenizer, prompts + heldout, fact["object"])
            if gold is None:
                continue
            negatives = {continuation_id(tokenizer, prompts, candidate)
                         for candidate in fact.get("negative_candidates", [])}
            negatives.difference_update({None, gold})
            states, embeddings, tops, lens_tops, entropy, gold_curves, base_curves, behavior = (
                representations(model, tokenizer, prompts + heldout, batch_size, gold, sorted(negatives)))
            index = len(manifest)
            filename = f"{name}-{index:06d}.pt"
            torch.save(dict(states=states[:len(prompts)], embeddings=embeddings[:len(prompts)],
                            lens_tops=lens_tops[:len(prompts)], entropy=entropy[:len(prompts)],
                            lens_gold_probability=gold_curves[:len(prompts)],
                            lens_base_probability=base_curves[:len(prompts)]),
                       directory / filename)
            manifest.append(dict(fact, paraphrases=prompts, behavior_prompts=heldout,
                                 gold=gold, negatives=sorted(negatives), file=filename,
                                 base=tops.tolist()[:len(prompts)],
                                 heldout=tops.tolist()[len(prompts):],
                                 behavior=behavior,
                                 base_text=tokenizer.batch_decode(tops[:, None]),
                                 lens_correct=(lens_tops[:len(prompts)] == gold).tolist()))
        if not manifest:
            raise ValueError(f"No single-token answers survived in {name}.")
        write_jsonl(directory / f"{name}.jsonl", manifest)
        print(f"{name}: cached {len(manifest)}/{len(facts)} facts", flush=True)
