"""Render TRACE-K Compact result figures from a benchmark JSON report."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "random_node": "Random nodes",
    "layer_importance": "Layer importance",
    "paraphrase_selection": "Paraphrase selection",
    "spectral": "Spectral",
    "similarity": "Activation similarity",
    "signal_aware": "Signal-aware",
    "prediction_aware": "Prediction-aware",
}
COLORS = ("#666666", "#7067a6", "#b3422d", "#1672aa", "#007864", "#b46a05", "#b33b77")


def plot(report, destination):
    test = report["test"]
    series = {name: [] for name in LABELS}
    for key in report["selected"]:
        method, reduction, _ = key.split("/")
        series[method].append((float(reduction) * 100, test[key]))
    for points in series.values():
        points.sort()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.6), layout="constrained", sharex=True)
    figure.suptitle(f"TRACE-K Compact | Llama 3.2 1B | {report['facts']['test']:,} held-out facts",
                    fontsize=15, fontweight="bold")
    panels = [
        ("accuracy", "Gold next-token accuracy (%)", 100),
        ("top1_agreement", "Full-probe top-1 agreement (%)", 100),
        ("signal_relative_sse", "Relative activation reconstruction error", 1),
        ("spectral_relative_energy_mse", "Relative spectral energy error", 1),
    ]
    for axis, (metric, ylabel, scale) in zip(axes.flat, panels, strict=True):
        for (name, points), color in zip(series.items(), COLORS, strict=True):
            axis.plot([reduction for reduction, _ in points],
                      [scale * row[metric] for _, row in points], "o-", color=color,
                      linewidth=1.8, markersize=4, label=LABELS[name])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.24)
        axis.set_xticks([25, 50, 75, 87.5], ["25", "50", "75", "87.5"])
    axes[0, 0].axhline(100 * test["full"]["accuracy"], color="#1a2632", linestyle="--",
                       linewidth=1.2, label="Full residual R-GCN")
    for axis in axes[-1]:
        axis.set_xlabel("Nodes removed (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def plot_systems(report, destination):
    full = report["test"]["full"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    figure.suptitle("TRACE-K Compact | Downstream GPU and storage costs", fontsize=15, fontweight="bold")
    for (name, label), color in zip(LABELS.items(), COLORS, strict=True):
        points = sorted((float(key.split("/")[1]) * 100, report["test"][key])
                        for key in report["selected"] if key.startswith(f"{name}/"))
        reductions = [reduction for reduction, _ in points]
        axes[0].plot(reductions, [(row["graph_bytes"] + row["final_bytes"]) / 1024
                                  for _, row in points], "o-", color=color, label=label, markersize=4)
        axes[1].plot(reductions, [row["inference_ms"] for _, row in points],
                     "o-", color=color, markersize=4)
        axes[2].plot(reductions, [row["compression_ms"] for _, row in points],
                     "o-", color=color, markersize=4)
    axes[0].axhline((full["graph_bytes"] + full["final_bytes"]) / 1024, linestyle="--",
                    color="#1a2632", label="Full residual R-GCN")
    axes[1].axhline(full["inference_ms"], linestyle="--", color="#1a2632")
    for axis, title in zip(axes, ("Payload with final states (KiB / fact)",
                                 "Probe forward (ms / fact)",
                                 "One-time construction (ms / fact)"), strict=True):
        axis.set_ylabel(title)
        axis.set_xlabel("Nodes removed (%)")
        axis.set_xticks([25, 50, 75, 87.5], ["25", "50", "75", "87.5"])
        axis.grid(axis="y", alpha=.24)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("figure", type=Path)
    parser.add_argument("--systems-figure", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    plot(report, args.figure)
    if args.systems_figure:
        plot_systems(report, args.systems_figure)


if __name__ == "__main__":
    main()
