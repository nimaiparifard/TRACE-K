"""Disjoint graph batches and view/layer ablations."""

import hashlib
import random
from functools import lru_cache

import torch
from torch.nn.functional import cosine_similarity, normalize
from torch.utils.data import Dataset

from .data import read_jsonl


class Activations(Dataset):
    def __init__(self, path, split):
        self.path = path
        self.rows = read_jsonl(path / f"{split}.jsonl")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return row, torch.load(self.path / row["file"], weights_only=True)


def rewire(edges, seed):
    """Shuffle endpoints within each relation while preserving in- and out-degrees."""
    rng = random.Random(seed)
    randomized = edges.copy()
    for relation in range(3):
        indices = [index for index, edge in enumerate(edges) if edge[2] == relation]
        if len(indices) < 2:
            continue
        occupied = {(edges[index][0], edges[index][1]) for index in indices}
        for _ in range(3 * len(indices)):
            first, second = rng.sample(indices, 2)
            a, b, kind, weight_first = randomized[first]
            c, d, _, weight_second = randomized[second]
            if a == d or c == b or (a, d) in occupied or (c, b) in occupied:
                continue
            occupied.difference_update(((a, b), (c, d)))
            occupied.update(((a, d), (c, b)))
            randomized[first] = a, d, kind, weight_first
            randomized[second] = c, b, kind, weight_second
    return randomized


@lru_cache(maxsize=256)
def canonical_edges(view_count, layers, inter_layer, cross_view):
    nodes = torch.arange(view_count * len(layers)).view(view_count, len(layers))
    edges = []
    if inter_layer:
        for position, (first, second) in enumerate(zip(layers, layers[1:])):
            if second == first + 1:
                for a, b in zip(nodes[:, position].tolist(), nodes[:, position + 1].tolist()):
                    edges.extend([(a, b, 0, 1.), (b, a, 0, 1.)])
    if cross_view:
        for column in nodes.T.tolist():
            edges.extend((a, b, 1, 1.) for a in column for b in column if a != b)
    return tuple(edges), torch.tensor(edges, dtype=torch.float32).reshape(-1, 4)


def graph(item, target, options, training=False, excluded_layer=None):
    row, cache = item
    states = cache["states"].float()
    views, depth, _ = states.shape
    selected = list(range(views)) if options["mode"] == "multi" else [target]
    dropout = options.get("view_dropout", 0) if training else 0
    selected = [view for view in selected if view == target or torch.rand(()) >= dropout]
    layers = list(range(depth))
    if isinstance(options.get("layers"), int):
        layers = [layers[options["layers"]]]
    elif options.get("layers") == "last":
        layers = layers[-1:]
    elif options.get("layers") == "middle":
        layers = layers[depth // 4:max(depth // 4 + 1, (3 * depth + 3) // 4)]
    layers = [layer for layer in layers if layer != excluded_layer]
    if not layers:
        raise ValueError("A graph must contain at least one layer.")
    x = states[selected][:, layers]
    if options.get("features") == "embedding":
        x = cache["embeddings"][selected, None].float().expand_as(x)
    elif options.get("features") == "stats":
        previous = torch.zeros_like(states[..., 0])
        previous[:, 1:] = cosine_similarity(states[:, 1:], states[:, :-1], dim=-1)
        positions = torch.arange(depth).expand(views, -1) / max(depth - 1, 1)
        stats = torch.stack([positions, states.norm(dim=-1), (states.abs() < 1e-3).float().mean(-1),
                             previous, cache["entropy"]], dim=-1)
        x = torch.cat([x, stats[selected][:, layers]], dim=-1)
    x = x.flatten(0, 1)
    edges, edge = canonical_edges(len(selected), tuple(layers), options.get("inter_layer", True),
                                  options.get("cross_view", True))
    anchor = (selected.index(target) + 1) * len(layers) - 1
    if not (options.get("similarity") or options.get("random_edges")):
        return x, edge, anchor
    edges = list(edges)
    if options.get("similarity", False) and len(x) > 1:
        unit = normalize(states[selected][:, layers].flatten(0, 1))
        similarities = unit @ unit.T
        similarities.fill_diagonal_(-torch.inf)
        similarity_values, neighbors = similarities.topk(min(3, len(x) - 1), dim=-1)
        for destination, (sources, values) in enumerate(zip(neighbors.tolist(), similarity_values.tolist())):
            edges.extend((src, destination, 2, max(0., value))
                         for src, value in zip(sources, values) if value > 0)
    if options.get("random_edges", False):
        digest = hashlib.sha256(f"{options.get('seed', 0)}:{row['fact_id']}:{target}".encode()).digest()
        edges = rewire(edges, int.from_bytes(digest[:8], "little"))
    edge = torch.tensor(edges, dtype=torch.float32).reshape(-1, 4)
    return x, edge, anchor


def collate(graphs, device):
    features, indices, weights, groups, anchors = [], [], [], [], []
    offset = 0
    for group, (x, edge, anchor) in enumerate(graphs):
        features.append(x)
        ids = edge[:, :3].long()
        ids[:, :2] += offset
        indices.append(ids)
        weights.append(edge[:, 3])
        groups.append(torch.full((len(x),), group, dtype=torch.long))
        anchors.append(anchor + offset)
        offset += len(x)
    return tuple(tensor.to(device) for tensor in
                 (torch.cat(features), torch.cat(indices), torch.cat(weights),
                  torch.cat(groups), torch.tensor(anchors)))
