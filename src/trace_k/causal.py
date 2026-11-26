"""Local attribution and matched interventions on the frozen decoder."""

import math
import random

import torch

from .llm import inputs_for


def activation_margin(model, inputs, position, layer, gold, foil, edit=None, gradient=False):
    module = model.model.layers[layer].mlp.down_proj
    captured = {}

    def hook(_module, args):
        original = args[0]
        activation = original[:, position].detach().clone()
        if edit is not None:
            activation = edit(activation)
        activation.requires_grad_(gradient)
        captured["activation"] = activation
        modified = original.clone()
        modified[:, position] = activation
        return (modified, *args[1:])

    handle = module.register_forward_pre_hook(hook)
    try:
        with torch.set_grad_enabled(gradient):
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            margin = logits[gold] - logits[foil]
            activation = captured["activation"]
            attribution = None
            if gradient:
                attribution = torch.autograd.grad(margin, activation)[0].detach().float()[0]
    finally:
        handle.remove()
    return logits.detach(), activation.detach().float()[0], attribution


def attribution(model, inputs, position, layer, gold, foil, steps):
    logits, activation, gradient = activation_margin(model, inputs, position, layer,
                                                     gold, foil, gradient=True)
    if steps > 1:
        gradients = [activation_margin(model, inputs, position, layer, gold, foil,
                                       edit=lambda values: values * ((index + .5) / steps),
                                       gradient=True)[2] for index in range(steps)]
        gradient = torch.stack(gradients).mean(0)
    return logits, activation, activation * gradient


def interventions(model, tokenizer, facts, layer, neurons, gains, steps, seed):
    rng = random.Random(seed)
    depth = len(model.model.layers)
    if depth < 2 or not 0 <= layer < depth:
        raise ValueError("Select a valid intervention layer in a decoder with at least two layers.")
    for fact_index, fact in enumerate(facts):
        gold = fact["gold"]
        for index, prompt in enumerate(fact["paraphrases"]):
            inputs, positions = inputs_for(tokenizer, [prompt], model.device)
            position = positions.item()
            with torch.no_grad():
                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            base = logits.argmax().item()
            foil = logits.masked_fill(torch.arange(len(logits), device=model.device) == gold,
                                       -torch.inf).argmax().item()
            _, activation, scores = attribution(model, inputs, position, layer, gold, foil, steps)
            count = min(neurons, len(activation) // 2)
            selected = scores.abs().topk(count).indices.tolist()
            available = sorted(set(range(len(activation))) - set(selected))
            random_indices = rng.sample(available, count)
            magnitude = activation.abs().cpu().tolist()
            matched = []
            for neuron in selected:
                nearest = min(available, key=lambda other: abs(magnitude[neuron] - magnitude[other]))
                matched.append(nearest)
                available.remove(nearest)
            other = rng.choice([candidate for candidate in range(depth) if candidate != layer])
            other_width = model.model.layers[other].mlp.down_proj.in_features
            other_facts = facts[fact_index + 1:] + facts[:fact_index]
            unrelated = next((candidate for candidate in other_facts
                              if candidate["gold"] != gold
                              and candidate["subject"] != fact["subject"]), None)
            controls = [("attributed", layer, selected), ("random", layer, random_indices),
                        ("activation_matched", layer, matched),
                        ("other_layer", other, rng.sample(range(other_width), min(count, other_width)))]
            cases = [controls]
            if unrelated:
                cases.append([("unrelated_fact", layer, selected)])
            for case in cases:
                is_unrelated = case[0][0] == "unrelated_fact"
                current_inputs, current_position, current_gold = inputs, position, gold
                current_logits, current_foil = logits, foil
                current_id, current_target = fact["fact_id"], index
                if is_unrelated:
                    current_id, current_target = unrelated["fact_id"], 0
                    current_inputs, last = inputs_for(tokenizer, unrelated["paraphrases"][:1], model.device)
                    current_position, current_gold = last.item(), unrelated["gold"]
                    with torch.no_grad():
                        current_logits = model(**current_inputs, use_cache=False,
                                               logits_to_keep=1).logits[0, -1].float()
                    current_foil = current_logits.masked_fill(
                        torch.arange(len(current_logits), device=model.device) == current_gold,
                        -torch.inf).argmax().item()
                before = (current_logits[current_gold] - current_logits[current_foil]).item()
                before_top = current_logits.argmax().item() if is_unrelated else base
                for name, chosen_layer, indices in case:
                    for gain in gains:
                        if not math.isfinite(gain) or gain < 0:
                            raise ValueError("Gains must be finite and nonnegative.")

                        def scale(values):
                            values[:, indices] *= gain
                            return values

                        modified, _, _ = activation_margin(
                            model, current_inputs, current_position, chosen_layer,
                            current_gold, current_foil, edit=scale)
                        after = (modified[current_gold] - modified[current_foil]).item()
                        yield dict(fact_id=current_id, target=current_target, source_fact_id=fact["fact_id"],
                                   control=name, layer=chosen_layer, neurons=indices,
                                   gain=gain, gold=current_gold, foil=current_foil,
                                   before=before_top, after=modified.argmax().item(),
                                   before_margin=before, after_margin=after, delta_margin=after - before)


def summarize_interventions(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["control"], row["gain"]), []).append(row)
    result = []
    for (control, gain), group in sorted(grouped.items()):
        failed = [row for row in group if row["before"] != row["gold"]]
        correct = [row for row in group if row["before"] == row["gold"]]
        result.append(dict(control=control, gain=gain, count=len(group),
                           mean_delta_margin=sum(row["delta_margin"] for row in group) / len(group),
                           correction_rate=(sum(row["after"] == row["gold"] for row in failed)
                                            / len(failed) if failed else None),
                           regression_rate=(sum(row["after"] != row["gold"] for row in correct)
                                            / len(correct) if correct else None)))
    return result
