# TRACE-K follow-up experiments

The full Llama 3.2 1B ParaRel cache contains 4,569 training, 979 validation, and 976 test facts. The test set has 7,808 graph prompts and 976 additional behavior prompts. The native frozen decoder scored 15.65% top-1 on the graph prompts. Confidence intervals below use 2,000 paired bootstrap resamples over facts. Each interval is conditional on the run's split, seed, and fitted models. [Machine-readable results](results.json)

## 1. Residual strength

The same residual R-GCN, trained with a 15% cap, was scored at all seven inference caps. The cap bounds the correction's L2 norm relative to the normalized target state. Zero reconstructs the output of the frozen head from the cached state in FP32. The cached state and original decoder use BF16, so the reconstructed argmax can differ from the native frozen argmax. The recovery and regression columns use native decoder predictions to define their denominators.

| Cap | Top-1 | Recovery | Regression | MRR |
| --- | ---: | ---: | ---: | ---: |
| 0% | 16.07% | 0.52% | 0.08% | 0.2637 |
| 1% | 19.21% | 4.24% | 0.08% | 0.2981 |
| 5% | 35.26% | 23.29% | 0.25% | 0.4567 |
| 10% | 50.54% | 41.63% | 1.47% | 0.5895 |
| 15% | 54.26% | 46.25% | 2.54% | 0.6323 |
| 25% | 53.10% | 45.38% | 5.32% | 0.6259 |
| 50% | 45.74% | 38.86% | 17.18% | 0.5599 |

The 1% cap increased top-1 by 3.14 percentage points over the reconstructed zero-cap baseline. At this checkpoint, measured accuracy peaked at the training cap and regression increased at larger caps.

## 2. All paraphrases wrong

The frozen decoder missed all nine measured paraphrases for 700 test facts. The additional behavior prompt determines subgroup membership. Probe metrics evaluate the eight graph targets per fact, for 5,600 targets.

| Residual R-GCN | Recovered targets | Recovery and top-1 | MRR |
| --- | ---: | ---: | ---: |
| Single-view | 2,308 | 41.21% | 0.5189 |
| Multi-view | 2,313 | 41.30% | 0.5207 |

The paired multi-view gain was +0.09 percentage points, with a 95% interval of [-1.18, +1.34]. The result concerns the nine recorded formulations of each fact.

## 3. Sign-aware intervention

Layer index 4 came from the existing validation graph-deletion calibration. On 64 validation facts, the sweep compared 8, 16, and 32 neurons and gains 1, 1.25, 1.5, and 2. Mean improvement in the gold-vs-foil logit margin selected 32 neurons and gain 2 among the nonidentity gains. One randomly selected prompt per fact was used on both validation and test. The test comprised 128 separate facts.

Gradient times activation selects the largest absolute attributions. Scaling each selected activation by `1 + (gain - 1) * sign(attribution)` amplifies positive contributions and suppresses negative ones. The foil is the strongest non-gold token before intervention. Gain 1 is the identity control. Selection uses each prompt's gold token.

| Gain-2 condition | Mean margin change | Corrections | Regressions |
| --- | ---: | ---: | ---: |
| Signed attribution | +1.354 | 9/111 | 0/17 |
| Unsigned attribution | +0.002 | 1/111 | 1/17 |
| Random neurons | -0.004 | 1/111 | 0/17 |
| Activation-matched neurons | -0.032 | 0/111 | 1/17 |
| Other layer | +0.003 | 1/111 | 0/17 |
| Directions and indices transferred to another fact | +0.007 | 2/111 | 1/17 |

## 4. Fair residual baselines

Trainable models use one epoch and a 15% cap. DeepSets, MLP, Transformer, and multi-view R-GCN see the same activation nodes. The pooled MLP uses the mean of projected nodes and the projected target anchor and approximately matches the R-GCN's parameter count. Mean-logit averages the eight reconstructed frozen final-layer logit vectors for each fact and uses zero trainable parameters.

| Method | Trainable parameters | Top-1 | Recovery | MRR |
| --- | ---: | ---: | ---: | ---: |
| Residual Transformer | 609,024 | 54.70% | 46.61% | 0.6334 |
| Residual R-GCN, multi-view | 1,035,904 | 54.26% | 46.25% | 0.6323 |
| Residual DeepSets | 559,488 | 53.18% | 44.85% | 0.6221 |
| Residual pooled MLP | 1,019,518 | 49.01% | 39.86% | 0.5835 |
| Mean-logit ensemble | 0 | 19.98% | 7.56% | 0.3251 |

Transformer minus R-GCN was +0.44 percentage points, with a 95% interval of [-0.15, +1.04]. Single-view residual R-GCN on the same overall test set reached 55.06% top-1.

## 5. Full-scale graph ablation

All three controls were rerun with a fixed checkpoint on the 976-fact test split. Node features and vocabulary readout stay fixed across graph conditions. Rewiring preserves directed in- and out-degrees within each edge type.

| Edge condition | Top-1 | Delta vs. full, percentage points | 95% interval |
| --- | ---: | ---: | ---: |
| Full | 54.26% | | |
| Remove cross-paraphrase edges | 54.19% | -0.08 | [-0.45, +0.28] |
| Remove inter-layer edges | 54.25% | -0.01 | [-0.23, +0.22] |
| Degree-preserving rewire | 54.02% | -0.24 | [-0.67, +0.22] |

All three intervals include zero. This protocol measures inference-time sensitivity of the fitted model to edge changes.

## 6. Synthetic exposure

SmolLM2-135M ran on 240 artificial facts, including 120 exposed to rank-32 LoRA training for 35 epochs. Exposure used three templates per exposed fact. A fourth template was held out of adapter training. Probe train, validation, and test split sizes were 168, 36, and 36. Of the test facts, 16 were exposed to LoRA and 20 were unexposed. The prespecified validation gate required at least 50% adapted top-1 on exposed facts and at least a 25-point gain over the base checkpoint. Validation top-1 on these prompts rose from 0% to 100%, so the probe comparison proceeded.

| Test metric | Base checkpoint | LoRA checkpoint |
| --- | ---: | ---: |
| Frozen exposed, trained templates | 0.00% | 100.00% |
| Frozen exposed, held-out template | 0.00% | 100.00% |
| Frozen unexposed, trained templates | 0.00% | 36.67% |
| Frozen unexposed, held-out template | 0.00% | 30.00% |
| Frozen exposure AUROC per fact | 0.5063 | 1.0000 |
| Residual probe exposure AUROC per fact | 0.3438 | 1.0000 |

Exposure scoring averages the known gold token's probability over the three graph prompts per fact. The adapted frozen decoder and the adapted probe both reached AUROC 1.0, so the experiment showed exposure detection without an additional probe benefit on this test.

## 7. Subject- and template-disjoint

Both splits were audited after extraction. Each started from a 1,500-fact sample before template and tokenizer filtering. Probe training used three epochs. The splits have different test populations and numbers of prompts per fact.

| Split | Test facts | Test prompts | Frozen top-1 | Residual R-GCN top-1 | Gain, percentage points (95% interval) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Subject-disjoint | 221 | 1,768 | 14.42% | 51.58% | +37.16 [32.41, 41.97] |
| Template-disjoint | 198 | 612 | 15.20% | 46.90% | +31.70 [25.69, 37.46] |

## Reproduce

Run from the repository root with the local ParaRel checkout and the Llama and SmolLM2 weights. The full cache and the three existing full-scale trained checkpoints are built by the first three commands in the [Compact protocol](../compact/protocol.md#reproduce) and its following `trace-k suite` command. For local paths, use `data/raw/pararel-main` for ParaRel and `data/raw/llama-3.2-1b` for the model. The sign-aware run uses the validation layer calibration stored in the tracked Compact results.

```powershell
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
foreach ($mode in "subject", "template") {
    trace-k split data/processed/facts.jsonl "data/splits/$mode" --mode $mode --limit 1500
    trace-k extract "data/splits/$mode" "data/processed/strength-$mode" --model-id data/raw/llama-3.2-1b --device cuda --batch-size 16
}
foreach ($section in "sweep", "fair", "causal", "graph", "synthetic", "disjoint") {
    python scripts/run_strengthening.py $section
}
```

The runner writes metrics and predictions under `runs/strengthening`. Bootstrap intervals describe the sampled test facts and fitted checkpoints. Runs use seed 42.
