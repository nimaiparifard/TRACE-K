"""Graph and graph-free probes with a fixed vocabulary projection."""

import torch
from torch import nn
from torch.nn import functional as F


class MessagePassing(nn.Module):
    def __init__(self, width, kind):
        super().__init__()
        self.kind = kind
        self.root = nn.Linear(width, width)
        self.neighbors = nn.ModuleList(nn.Linear(width, width, bias=False)
                                       for _ in range(3 if kind == "rgcn" else 1))
        self.norm = nn.LayerNorm(width)
        if kind == "gat":
            self.left = nn.Linear(width, width, bias=False)
            self.right = nn.Linear(width, width, bias=False)
            self.attention = nn.Linear(width, 1, bias=False)

    def forward(self, x, edges, weights):
        source, dest, relation = edges.T
        result = self.root(x)
        if self.kind == "gat":
            logits = self.attention(F.leaky_relu(self.left(x[source]) + self.right(x[dest]), .2))
            weights = weights * logits.squeeze(-1).clamp(-30, 30).exp()
        if self.kind == "gcn":
            degree = weights.new_ones(len(x)).index_add_(0, dest, weights)
            result = result / degree[:, None]
        for index, transform in enumerate(self.neighbors):
            mask = relation == index if self.kind == "rgcn" else torch.ones_like(relation, dtype=torch.bool)
            src, dst, scale = source[mask], dest[mask], weights[mask]
            if self.kind == "gcn":
                scale = scale / (degree[src] * degree[dst]).sqrt()
            else:
                totals = weights.new_zeros(len(x)).index_add_(0, dst, scale)
                scale = scale / totals[dst].clamp_min(1e-12)
            messages = transform(x)[src] * scale[:, None]
            result = result.index_add(0, dst, messages)
        return x + F.gelu(self.norm(result))


class Probe(nn.Module):
    def __init__(self, size, head, options, pca=None):
        super().__init__()
        width = options["width"]
        self.kind = options["model"]
        self.residual_limit = options.get("residual_limit", 0)
        if self.residual_limit and self.kind == "tuned":
            raise ValueError("Tuned Lens already uses a separate residual translator.")
        if self.residual_limit < 0:
            raise ValueError("The residual limit must be nonnegative.")
        self.register_buffer("unembedding", head["weight"].float(), persistent=False)
        bias = head["bias"]
        self.register_buffer("bias", None if bias is None else bias.float(), persistent=False)
        if pca is not None:
            self.register_buffer("mean", pca[0])
            self.register_buffer("components", pca[1])
            size = len(pca[1])
        if self.kind == "tuned" or self.residual_limit:
            self.register_buffer("norm_weight", head["norm_weight"].float(), persistent=False)
            self.norm_epsilon = head["norm_epsilon"]
        if self.kind == "tuned":
            self.translator = nn.Linear(size, head["weight"].shape[1])
            nn.init.zeros_(self.translator.weight)
            nn.init.zeros_(self.translator.bias)
            return
        self.project = nn.Linear(size, width)
        if self.kind in {"rgcn", "gcn", "sage", "gat"}:
            projection_width = options.get("projection_width", 256)
            self.project = nn.Sequential(nn.Linear(size, projection_width), nn.GELU(),
                                         nn.Linear(projection_width, width))
            self.layers = nn.ModuleList(MessagePassing(width, self.kind) for _ in range(3))
        if self.kind in {"mlp", "deepsets"}:
            inner = width
            if options.get("match_rgcn"):
                projection_width = options.get("projection_width", 256)
                extra_projection = ((size + 1) * projection_width
                                    + (projection_width + 1) * width - (size + 1) * width)
                rgcn_layers = 3 * (4 * width * width + 3 * width)
                inner = round((extra_projection + rgcn_layers - width) / (2 * width + 1))
            self.mlp = nn.Sequential(nn.Linear(width, inner), nn.GELU(), nn.Linear(inner, width))
        if self.kind == "transformer":
            self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        if self.kind in {"rgcn", "gcn", "sage", "gat", "transformer"}:
            self.query = nn.Linear(width, width)
        self.output = nn.Linear(width, head["weight"].shape[1])
        if self.residual_limit:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, batch, final=None):
        return F.linear(self.vocabulary_input(batch, final), self.unembedding, self.bias)

    def vocabulary_input(self, batch, final=None, return_pool=False, shared_graph=False):
        x, edges, weights, groups, anchors = batch
        if self.kind == "tuned":
            residual = x[anchors]
            return self.normalize(residual + self.translator(residual))
        if hasattr(self, "components"):
            x = F.linear(x - self.mean, self.components)
        x = self.project(x)
        if hasattr(self, "layers"):
            for layer in self.layers:
                x = layer(x, edges, weights)
        if self.kind == "linear":
            pooled = x[anchors]
        elif self.kind == "mlp":
            pooled = self.mlp(x[anchors])
        elif self.kind == "deepsets":
            x = self.mlp(x)
            sums = x.new_zeros(len(anchors), x.size(1)).index_add_(0, groups, x)
            pooled = sums / torch.bincount(groups, minlength=len(anchors))[:, None]
        elif shared_graph:
            scores = self.query(x[anchors]) @ x.T / x.size(-1) ** .5
            pooled = scores.softmax(-1) @ x
        else:
            if self.kind == "transformer":
                x = torch.cat([self.attention(nodes[None], nodes[None], nodes[None],
                                              need_weights=False)[0][0] + nodes
                               for nodes in (x[groups == i] for i in range(len(anchors)))])
            scores = (self.query(x[anchors])[groups] * x).sum(-1) / x.size(-1) ** .5
            maxima = scores.new_full((len(anchors),), -torch.inf)
            maxima.scatter_reduce_(0, groups, scores.detach(), reduce="amax")
            attention = (scores - maxima[groups]).exp()
            totals = scores.new_zeros(len(anchors)).index_add_(0, groups, attention)
            pooled = x.new_zeros(len(anchors), x.size(1)).index_add_(
                0, groups, x * (attention / totals[groups])[:, None])
        prediction = self.output(pooled)
        if self.residual_limit:
            if final is None:
                raise ValueError("The residual probe needs the original final-layer state.")
            base = self.normalize(final)
            cap = self.residual_limit * base.norm(dim=-1, keepdim=True)
            scale = (cap / prediction.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1)
            prediction = base + prediction * scale
        return (prediction, pooled) if return_pool else prediction

    def lens(self, residual):
        return F.linear(self.normalize(residual), self.unembedding, self.bias)

    def normalize(self, residual):
        normalized = residual * torch.rsqrt(
            residual.square().mean(-1, keepdim=True) + self.norm_epsilon)
        return normalized * self.norm_weight
