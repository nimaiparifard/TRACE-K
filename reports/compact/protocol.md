# TRACE-K Compact

Can we compress LLM activation graphs while preserving factual information and GNN predictions? Here, factual information is measured by exact gold-token accuracy, full-vocabulary MRR, and recovery on targets missed by the frozen decoder. Prediction preservation is measured by agreement with the full residual R-GCN and by full-vocabulary KL divergence.

## Protocol

Each Llama graph has eight paraphrases and 16 decoder layers. It contains 128 activation nodes and 1,136 directed, typed edges. Every fact is evaluated on its eight target paraphrases. The full and compact probes share a single fixed checkpoint, the original final-layer residual for each target, the cached RMSNorm parameters, and the frozen vocabulary head. Each target's readout anchor maps to the representative of its original final-layer node. Full and compact R-GCN inference both share message passing across the eight targets.

| Method | Rule |
|---|---|
| Random node | Seeded uniform node ordering per fact |
| Layer importance | Keep layers by mean prediction KL after single-layer removal on a seeded validation subset |
| Paraphrase selection | Keep diverse complete view trajectories using farthest-first ordering over normalized activations |
| Spectral | Cluster in the first eight nonconstant modes of the symmetrized combinatorial Laplacian |
| Similarity | Cluster by cosine geometry over normalized activations |
| Signal-aware | Cluster in concatenated activation and spectral coordinates |
| Prediction-aware | Add gradient salience from the full probe's predicted top-two logit margin to the signal-aware geometry and prioritize salient representatives |

The sampling methods keep induced typed edges. Coarsening takes the mean activation in each cluster and sums edge weights separately by relation, including contracted edges as typed self-loops. The four node budgets use prefixes of the same representative ordering for a given method and geometry coefficient. For reconstruction, dropped sampling nodes map to the closest retained node in activation space, while merged nodes map to their own cluster mean.

The signal diagnostic is squared Frobenius reconstruction error divided by the squared Frobenius norm of the full activation matrix. The spectral diagnostic compares the Dirichlet energy of the selected full-graph modes against the energy of their cluster-averaged counterparts on the compact graph. It uses the mean squared relative error over those modes. The prediction diagnostic is the mean KL divergence from the full distribution to the compact distribution. The weighted sum of those three diagnostics chooses the geometry coefficient of each signal- and prediction-aware method at each budget on validation. The reported run gives each diagnostic a coefficient of one and searches geometry coefficients 0.1 and 1.0. The test loop evaluates only the selected configurations. Cluster construction uses no gold answers. Prediction-aware construction runs the full probe as a teacher and differentiates its margin at construction time.

The benchmark uses 2,000 paired bootstrap draws over facts, with all eight prompts resampled together. The intervals are pointwise and conditional on the split, seed, and fitted checkpoint. Gold accuracy and full-probe agreement answer different questions. A compressed graph can improve gold accuracy while changing the teacher's predictions.

## Prediction preservation

Let $s$ be the original normalized final state and $U$ the fixed vocabulary matrix. With corrections $\delta$ and $\delta_c$, the shared head gives

$$
\|z-z_c\|_2 \leq \|U\|_2\|\delta-\delta_c\|_2,\qquad
\|z-z_c\|_\infty \leq \max_i\|U_{i,:}\|_2\|\delta-\delta_c\|_2.
$$

For a strictly positive full-model margin $m$ between the top two logits, any upper bound below $m/2$ certifies the same argmax. The report measures the observed logit perturbation, the bound from the measured correction error, and a bound from the pooled GNN difference. For the latter, the correction is the projection of $Wh+b$ onto an L2 ball whose radius depends only on $s$. This projection is nonexpansive, so $\|\delta-\delta_c\|_2 \le \|W\|_2\|h-h_c\|_2$.

The pooled error can be bounded further from graph structure. Let $P$ be the hard assignment matrix, $Y_\ell$ and $Y_{c,\ell}$ the full and compact node embeddings, and $E_\ell=\|Y_\ell-PY_{c,\ell}\|_F$. For relation $r$, let $A_r$ and $B_r$ be the receiver-normalized full and compact message operators used by the R-GCN, and let $W_r$ and $W_0$ be its learned neighbor and root maps at that layer. Then the residual block satisfies

$$
E_{\ell+1}\le
\left[1+L_\ell\left(\|W_0\|_2+\sum_r\|A_r\|_2\|W_r\|_2\right)\right]E_\ell
+L_\ell\sum_r\|A_rP-PB_r\|_2\|Y_{c,\ell}\|_F\|W_r\|_2.
$$

Here $L_\ell$ is a Lipschitz constant for row-wise LayerNorm followed by GELU. A conservative choice is $2\|\gamma_\ell\|_\infty/\sqrt{\epsilon_\ell}$. The affine bias terms cancel after lifting because each row of $P$ sums to one. The initial error is bounded by the Lipschitz constant of the probe input projector times $\|X-PX_c\|_F$. The term $A_rP-PB_r$ tracks relation-specific distortion from node removal or contraction even when reconstruction error is small.

For attention pooling, lift the compact attention $\alpha_c$ to $\tilde\alpha_c$ by distributing each cluster's attention evenly over its original nodes. The lifted compact pool is exactly the compact pool. The triangle inequality gives $\|h-h_c\|_2\le E_L+\max_j\|Y_{c,L}[j]\|_2\,\|\alpha-\tilde\alpha_c\|_1$.

Let $q_c$ be the compact anchor query, $Q$ the query matrix, $d$ the embedding width, and $n_{\min},n_{\max}$ the smallest and largest cluster sizes. Softmax is 1-Lipschitz from $\ell_\infty$ to $\ell_1$ because its directional derivative is the probability-weighted mean absolute deviation of the input direction. Lifted compact attention equals softmax over the lifted compact scores with $\log$ cluster size subtracted. Softmax is invariant to a shared shift, so centering the log cluster-size range gives

$$
\|\alpha-\tilde\alpha_c\|_1 \le
\min\left\{2,\,
\frac{(\|Q\|_2\max_i\|Y_L[i]\|_2+\|q_c\|_2)E_L}{\sqrt{d}}
+\frac12\log\frac{n_{\max}}{n_{\min}}\right\}.
$$

This recurrence is a structural explanation and can be loose. The benchmark numerically evaluates the correction and pooled bounds above.

## Measurement

The compact run reports graph tensor payload bytes in FP32, including typed weighted edges and anchor indices. The final residual side information is reported separately at the cache's source dtype. Storage counts tensor payloads, excluding archive metadata and the shared head/model. Batch time includes packing and device transfer of the graph. Probe inference includes the R-GCN, attention readout, and full-vocabulary projection, with CUDA synchronization. GPU peak allocation is process-local and includes resident tensors. Incremental GPU allocation is measured above the already-packed graph and model. Construction time includes ranking/assignment and quotient or induced edges. It charges the shared nested representative plan to every budget and includes the full teacher pass and salience computation for Prediction-aware. Compact graph payload storage refers to a graph after construction.

The extraction report separately times frozen-decoder representation collection and total extraction wall time, then records peak allocated CUDA memory. Representation collection includes tokenization, forward passes, activation hooks, Logit Lens scoring, and copies to host memory. Every compact method consumes cached LLM activations, so its probe latency and graph storage figures quantify downstream processing. Extraction profiling starts after the LLM is loaded.

## Reproduce

```powershell
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
trace-k prepare data/raw/pararel data/processed/facts.jsonl --views 8 --behavior-views 1
trace-k split data/processed/facts.jsonl data/splits/compact-full --mode fact
trace-k extract data/splits/compact-full data/processed/llama-compact-full --model-id meta-llama/Llama-3.2-1B --device cuda --batch-size 8
trace-k suite data/processed/llama-compact-full runs/compact-full-controls --device cuda --epochs 1 --batch-size 8 --variants residual_rgcn residual_deepsets residual_transformer
trace-k compact data/processed/llama-compact-full runs/compact-full-controls/residual_rgcn/probe.pt runs/compact-full.json --device cuda --ablations
trace-k compare runs/compact-full-controls/residual_rgcn runs/compact-full-paired.json runs/compact-full-controls/residual_deepsets runs/compact-full-controls/residual_transformer
python -m pip install -e ".[compact-report]"
python scripts/plot_compact.py runs/compact-full.json reports/compact/full-curves.png --systems-figure reports/compact/full-systems.png
```

The full fixed-checkpoint ablations remove each edge relation, perform degree-preserving rewiring, restrict graphs to the last or middle layers, and remove each layer individually. Each ablation applies to every held-out fact. The same target final-layer residual reaches the shared vocabulary head in all conditions.

To reproduce the three-epoch benchmark, use a separate split and activation directory with `--limit 1500`. Train the variants `residual_rgcn residual_deepsets residual_transformer residual_no_cross_view residual_no_inter_layer residual_random_graph residual_last_layer residual_middle_layers` with `--epochs 3`. Run Compact against that split's residual R-GCN checkpoint.
