"""Browse TRACE-K reports and per-prompt measurements."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from trace_k.data import read_jsonl
from trace_k.graph import Activations, graph


st.set_page_config(page_title="TRACE-K Research Explorer", layout="wide")
st.title("TRACE-K Research Explorer")
root = Path(st.sidebar.text_input("Results directory", "runs"))
reports = sorted(root.rglob("metrics.json"))
if not reports:
    st.info("Train a probe and select its results directory.")
    st.stop()

report_path = st.sidebar.selectbox("Experiment", reports, format_func=lambda path: str(path.parent))
report = json.loads(report_path.read_text(encoding="utf-8"))
rows = read_jsonl(report_path.with_name("predictions.jsonl"))
metrics = report["test"]
for column, key in zip(st.columns(4), ("accuracy", "recovery_rate", "mrr", "consistency")):
    value = metrics[key]
    column.metric(key.replace("_", " ").title(), "N/A" if value is None else f"{value:.3f}")
comparison = []
for path in reports:
    result = json.loads(path.read_text(encoding="utf-8"))
    source = result.get("source", {})
    test = result["test"]
    comparison.append(dict(experiment=str(path.parent.relative_to(root)),
                           model=source.get("model_id", ""),
                           split=source.get("split", {}).get("mode", ""),
                           test_prompts=test["count"], accuracy=test["accuracy"],
                           recovery_rate=test["recovery_rate"], mrr=test["mrr"]))
st.dataframe(comparison, hide_index=True)
if "paired_test" in report:
    with st.expander("Performance by answer exposure"):
        rows_by_exposure = []
        for group, label in (("all", "All"), ("seen_answer", "Seen in training"),
                             ("unseen_answer", "Unseen in training")):
            values = report["paired_test"][group]
            if values:
                interval = values["accuracy_delta_ci95"]
                rows_by_exposure.append({
                    "Answers": label, "Facts": values["facts"], "Prompts": values["prompts"],
                    "Frozen accuracy": f"{values['frozen_accuracy']:.1%}",
                    "Probe accuracy": f"{values['accuracy']:.1%}",
                    "Accuracy change (pp)": f"{100 * values['accuracy_delta']:+.1f}",
                    "95% interval (pp)": f"[{100 * interval[0]:+.1f}, {100 * interval[1]:+.1f}]"
                    if interval is not None else "N/A",
                })
        st.dataframe(rows_by_exposure, hide_index=True)
st.plotly_chart(px.line(y=report["logit_lens_accuracy_by_layer"],
                        labels={"x": "Layer", "y": "Logit lens accuracy"}))

evidence = st.selectbox("Evidence", ["All"] + sorted({row["evidence"] for row in rows}))
choices = [row for row in rows if evidence == "All" or row["evidence"] == evidence]
selected = st.selectbox("Prompt", choices, format_func=lambda row: row["prompt"])
st.write({"Expected answer": selected["answer"], "LLM prediction": selected["base_text"],
          "Probe prediction": selected.get("prediction_text", selected["prediction"]),
          "Gold rank": selected["rank"], "Evidence": selected["evidence"]})

cache = Path(st.sidebar.text_input("Activation cache", value=report.get("cache", "data/processed/cache"),
                                   key=f"cache:{report_path}"))
if (cache / "test.jsonl").is_file():
    dataset = Activations(cache, "test")
    indices = [index for index, row in enumerate(dataset.rows)
               if row["fact_id"] == selected["fact_id"]]
    if indices:
        fact, tensors = dataset[indices[0]]
        st.plotly_chart(px.imshow(tensors["entropy"].tolist(),
                                  labels={"x": "Layer", "y": "View", "color": "Entropy"},
                                  aspect="auto", title="Logit lens entropy"))
        options = report["options"]
        target = selected["target"]
        if "lens_gold_probability" in tensors:
            curves = {"Expected token": tensors["lens_gold_probability"][target].tolist()}
            if selected["base"] != selected["gold"]:
                curves["LLM prediction"] = tensors["lens_base_probability"][target].tolist()
            st.plotly_chart(px.line(pd.DataFrame(curves), markers=True,
                                    labels={"index": "Layer", "value": "Probability",
                                            "variable": "Token"},
                                    title="Layer-wise Logit Lens probabilities"))
        _, edges, _ = graph((fact, tensors), target, options)
        depth = tensors["states"].shape[1]
        layers = list(range(depth))
        if options.get("layers") == "last":
            layers = layers[-1:]
        elif options.get("layers") == "middle":
            layers = layers[depth // 4:max(depth // 4 + 1, (3 * depth + 3) // 4)]
        elif isinstance(options.get("layers"), int):
            layers = [options["layers"]]
        views = [target] if options["mode"] == "single" else range(len(fact["paraphrases"]))
        coordinates = [(layer, view) for view in views for layer in layers]
        figure = go.Figure()
        for relation, title, color in [(0, "Inter-layer", "#64748b"),
                                       (1, "Cross-view", "#38bdf8"),
                                       (2, "Similarity", "#c084fc")]:
            x, y = [], []
            for a, b, kind, _ in edges.tolist():
                if kind == relation:
                    start, end = coordinates[int(a)], coordinates[int(b)]
                    x.extend([start[0], end[0], None])
                    y.extend([start[1], end[1], None])
            figure.add_scatter(x=x, y=y, mode="lines", name=title,
                               line={"color": color, "width": 1}, opacity=.25)
        figure.add_scatter(x=[point[0] for point in coordinates],
                           y=[point[1] for point in coordinates], mode="markers", name="Residuals",
                           text=[fact["paraphrases"][view] for _, view in coordinates],
                           marker={"size": 9, "color": ["#f97316" if view == target else "#38bdf8"
                                                        for _, view in coordinates]})
        figure.update_layout(title="Activation graph", xaxis_title="Layer", yaxis_title="View")
        st.plotly_chart(figure)

causal = Path(st.sidebar.text_input("Intervention results", "runs/causal.jsonl"))
if causal.is_file():
    interventions = pd.DataFrame(row for row in read_jsonl(causal)
                                 if row["fact_id"] == selected["fact_id"]
                                 and row["target"] == selected["target"])
    if not interventions.empty:
        st.plotly_chart(px.scatter(interventions, x="gain", y="delta_margin", color="control",
                                    hover_data=["layer", "before", "after"],
                                    title="Neuron intervention effects"))
        st.dataframe(interventions, hide_index=True)
