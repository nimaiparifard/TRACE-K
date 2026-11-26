"""Controlled synthetic fact exposure via a separate LoRA checkpoint."""

import torch

from .data import read_jsonl
from .llm import continuation_id, inputs_for, load_model


def adapt(facts, out, options, epochs, batch_size, training_views=1, rank=8, learning_rate=.001):
    from peft import LoraConfig, get_peft_model

    model, tokenizer = load_model(*options)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=rank, lora_alpha=2 * rank,
                                           target_modules=["q_proj", "v_proj"]))
    records = []
    for row in read_jsonl(facts):
        if row.get("exposed") is True:
            for view in row["paraphrases"][:training_views]:
                prompt = view.rstrip()
                token = continuation_id(tokenizer, [prompt], row["object"])
                if token is not None:
                    records.append((prompt, token))
    if not records:
        raise ValueError("No exposed single-token facts available.")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        loss_sum = 0.
        indices = torch.randperm(len(records)).tolist()
        for start in range(0, len(indices), batch_size):
            prompts, tokens = zip(*(records[i] for i in indices[start:start + batch_size]))
            inputs, _ = inputs_for(tokenizer, list(prompts), model.device)
            targets = torch.tensor(tokens, device=model.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(targets)
        print(f"LoRA epoch {epoch + 1}/{epochs} | loss {loss_sum / len(records):.4f}", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(out)
