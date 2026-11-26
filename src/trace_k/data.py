"""Dataset ingestion and grouped experimental partitions."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def pararel(root, views, behavior_views=0):
    root = Path(root)
    patterns = root / "data" / "pattern_data" / "graphs_json"
    triples_dir = root / "data" / "trex_lms_vocab"
    if not patterns.is_dir() or not triples_dir.is_dir():
        raise ValueError("Provide the root of the official ParaRel checkout.")
    if behavior_views < 0:
        raise ValueError("Behavior view count must be nonnegative.")
    for pattern_file in sorted(patterns.glob("P*.jsonl")):
        relation = pattern_file.stem
        templates = sorted({row["pattern"] for row in read_jsonl(pattern_file)
                            if "[X]" in row["pattern"] and "[Y]" in row["pattern"]})
        prefixes = sorted({template.split("[Y]", 1)[0].rstrip() for template in templates
                           if "[X]" in template.split("[Y]", 1)[0]})
        required = views + behavior_views
        if len(prefixes) < required:
            continue
        triples_path = triples_dir / pattern_file.name
        if not triples_path.is_file():
            continue
        triples = read_jsonl(triples_path)
        objects = sorted({row["obj_label"] for row in triples})
        for row in triples:
            subject, answer = row["sub_label"], row["obj_label"]
            identity = hashlib.sha256(json.dumps([subject, relation, answer]).encode()).hexdigest()
            yield {
                "fact_id": identity,
                "subject": subject,
                "relation": relation,
                "object": answer,
                "paraphrases": [prefix.replace("[X]", subject) for prefix in prefixes[:views]],
                "templates": prefixes[:views],
                "behavior_prompts": [prefix.replace("[X]", subject)
                                     for prefix in prefixes[views:required]],
                "behavior_templates": prefixes[views:required],
                "negative_candidates": random.Random(identity).sample(
                    [other for other in objects if other != answer], min(32, len(objects) - 1)),
            }


def partitions(rows, mode, seed, limit=None):
    """Keep duplicates and all views of a fact in the same partition."""
    groups = defaultdict(list)
    owners = defaultdict(set)
    eligible = []
    for row in rows:
        identity = (row["subject"].strip().casefold(), row["relation"], row["object"].strip().casefold())
        for prompt in row["paraphrases"] + row.get("behavior_prompts", []):
            owners[prompt.strip().casefold()].add(identity)
    key = {"fact": "fact_id", "subject": "subject", "relation": "relation",
           "object": "object", "template": "fact_id"}[mode]
    seen = set()
    for row in rows:
        prompts = row["paraphrases"] + row.get("behavior_prompts", [])
        if any(len(owners[prompt.strip().casefold()]) > 1 for prompt in prompts):
            continue
        identity = (row["subject"].strip().casefold(), row["relation"], row["object"].strip().casefold())
        if identity not in seen:
            seen.add(identity)
            eligible.append(row)
    rng = random.Random(seed)
    rng.shuffle(eligible)
    for row in eligible[:limit]:
        groups[str(row[key]).strip().casefold()].append(row)
    keys = sorted(groups)
    rng.shuffle(keys)
    if len(keys) < 3:
        raise ValueError(f"The {mode} split requires at least three distinct groups.")
    train_end = min(len(keys) - 2, max(1, round(len(keys) * 0.70)))
    val_end = min(len(keys) - 1, train_end + max(1, round(len(keys) * 0.15)))
    split = {name: [row for key in group for row in groups[key]]
             for name, group in zip(("train", "val", "test"),
                                    (keys[:train_end], keys[train_end:val_end], keys[val_end:]))}
    if mode == "template":
        templates = sorted({template for row in rows
                            for template in row["templates"] + row.get("behavior_templates", [])})
        rng.shuffle(templates)
        ownership = {template: index % 3 for index, template in enumerate(templates)}
        for index, name in enumerate(split):
            selected = []
            for row in split[name]:
                pairs = [(prompt, template) for prompt, template in
                         zip(row["paraphrases"], row["templates"], strict=True)
                         if ownership[template] == index]
                if pairs:
                    prompts, used = zip(*pairs)
                    heldout = [(prompt, template) for prompt, template in zip(
                        row.get("behavior_prompts", []), row.get("behavior_templates", []), strict=True)
                        if ownership[template] == index]
                    selected.append(dict(row, paraphrases=list(prompts), templates=list(used),
                                         behavior_prompts=[prompt for prompt, _ in heldout],
                                         behavior_templates=[template for _, template in heldout]))
            split[name] = selected
    if any(not items for items in split.values()):
        raise ValueError(f"The {mode} split has an empty partition.")
    audit_partitions(split, mode)
    return split


def audit_partitions(split, mode="fact"):
    for name, rows in split.items():
        if len({str(row["fact_id"]).strip().casefold() for row in rows}) != len(rows):
            raise ValueError(f"Duplicate fact identifiers in {name}.")
    fields = ["fact_id", "fact", "prompts"]
    if mode != "fact":
        fields.append(mode)

    def values(rows, field):
        if field == "fact":
            return {tuple(str(row[key]).strip().casefold() for key in ("subject", "relation", "object"))
                    for row in rows}
        if field in {"prompts", "template"}:
            key = "paraphrases" if field == "prompts" else "templates"
            extra = "behavior_prompts" if field == "prompts" else "behavior_templates"
            return {value.strip().casefold() for row in rows
                    for value in row[key] + row.get(extra, [])}
        return {str(row[field]).strip().casefold() for row in rows}

    for field in fields:
        seen = set()
        for rows in split.values():
            current = values(rows, field)
            if seen & current:
                raise ValueError(f"Partition leakage detected for {field}.")
            seen.update(current)


def synthetic(objects, count, seed):
    """Balance randomly assigned labels and use opaque subject identifiers."""
    rng = random.Random(seed)
    objects = sorted(set(objects))
    if len(objects) < 2 or count < 3:
        raise ValueError("Synthetic data requires at least two labels and three facts.")
    labels = (objects * ((count + len(objects) - 1) // len(objects)))[:count]
    rng.shuffle(labels)
    exposed_indices = set()
    for answer in objects:
        assigned = [index for index, label in enumerate(labels) if label == answer]
        exposed_indices.update(rng.sample(assigned, len(assigned) // 2))
    templates = ["The capital of [X] is", "[X] has its capital in",
                 "The seat of government of [X] is", "The capital city of [X] is"]
    for index, answer in enumerate(labels):
        subject = f"Territory {rng.getrandbits(64):016x}"
        yield dict(fact_id=f"synthetic-{index}", subject=subject, relation="capital",
                   object=answer, exposed=index in exposed_indices, templates=templates,
                   paraphrases=[template.replace("[X]", subject) for template in templates],
                   negative_candidates=[other for other in objects if other != answer])
