"""Unsmoothed pilot figures; final experiment figures follow audited evaluation."""

import os
from pathlib import Path

from .acquire import ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json


def pilot_figure(ids, root=ROOT):
    root = Path(root)
    fig, axes = plt.subplots(len(ids), 2, figsize=(12, 3.1 * len(ids)), squeeze=False, constrained_layout=True)
    for row, rid in enumerate(ids):
        frame = pd.read_csv(root / f"data/cache/{rid}_measurements.csv")
        normalized = frame.inner_brow_height.notna().any()
        value = frame.inner_brow_height if normalized else frame.inner_brow_distance_px
        units = "height / fixed reference eye span" if normalized else "distance (relative pixel coordinates)"
        axes[row, 0].plot(frame.timestamp_s, frame.pitch_deg, color="#245c95", linewidth=1.3)
        axes[row, 1].plot(frame.timestamp_s, value, color="#a94b35", linewidth=1.3)
        for col in range(2):
            axes[row, col].set_xlabel("Time (seconds)")
            axes[row, col].grid(alpha=.2)
            axes[row, col].set_title(f"{rid} | " + ("estimated pitch" if col == 0 else "raw eyebrow proxy"), fontsize=10)
            reference = frame.loc[frame.reference_frame, "timestamp_s"]
            if len(reference):
                axes[row, col].axvspan(reference.min(), reference.max(), color="#7c9887", alpha=.15)
        axes[row, 0].set_ylabel("Pitch (degrees; uncalibrated model pose)")
        axes[row, 1].set_ylabel(units)
    fig.suptitle("Pilot extraction — raw measurements; candidate references; no learned correction", fontsize=13)
    destination = root / "results/figures/pilot_measurements.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)
    print(destination)
    return destination


COLORS = {"raw": "#343a40", "gam": "#bd622a", "mlp": "#187d86"}


def experiment_figures(root=ROOT, examples=None):
    """Export the frozen comparison without smoothing or selecting a test seed."""
    if examples is None or len(examples) != 3:
        raise ValueError("Pilot trajectories require three local recording IDs via --recording-ids; see docs/reproduction.md.")
    from .acquire import read_manifest
    from .evaluate import fit_model
    from .models import balanced_samples
    root = Path(root)
    run = json.loads((root / "results/face-landmarker/run_metadata.json").read_text())
    config = run["config"]
    rows = [r for r in read_manifest(root) if r["include"] == "true"]
    tables = {r["recording_id"]: pd.read_csv(root / f"data/cache/{r['recording_id']}_measurements.csv") for r in rows}
    metrics = pd.read_csv(root / "results/face-landmarker/recording_metrics.csv")
    pred = pd.read_csv(root / "results/face-landmarker/predictions.csv")
    annotations = pd.read_csv(root / "data/annotations.csv")
    out = root / "results/figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=200)
        fig.savefig(out / f"{name}.svg")
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10, 8.2), constrained_layout=True)
    for i, fold in enumerate(run["folds"]):
        train = [r for r in rows if r["recording_id"] in fold["neutral_training_ids"]]
        sample = balanced_samples([tables[r["recording_id"]] for r in train], config["training"]["max_samples_per_recording"])
        axes[i, 0].scatter(sample.relative_pitch_deg, sample.target, s=8, alpha=.25, color="#577b9e", label="Balanced training samples")
        for method in ["gam", "mlp"]:
            spec = fold["selected_models"][method]
            model = fit_model(method, spec["penalty"], train, tables, config, spec["seed"])
            grid = np.linspace(model.low, model.high, 240)
            correction, _ = model.correction(grid)
            axes[i, 0].plot(grid, correction, color=COLORS[method], lw=1.8, label=method.upper())
        test_ids = [r["recording_id"] for r in rows if r["distance"] == fold["held_out_distance"] and r["eyebrow_condition"] == "neutral"]
        for method in ["raw", "gam", "mlp"]:
            xs, ys = [], []
            for rid in test_ids:
                frame = pred[pred.recording_id == rid]
                raw = tables[rid]
                baseline = raw.loc[raw.reference_frame & raw.valid_geometry, "inner_brow_height"].median()
                mask = frame.supported_pose & frame.valid_geometry & ~frame.reference_frame
                xs.extend(frame.loc[mask, "relative_pitch_deg"])
                ys.extend(frame.loc[mask, method + "_height"] - baseline)
            axes[i, 1].scatter(xs, ys, s=7, alpha=.28, color=COLORS[method], label=method.upper())
        for j in range(2):
            axes[i, j].axhline(0, color="#999999", lw=.7)
            axes[i, j].set_title(f"Held out: {fold['held_out_distance']} | " + ("training relationship" if j == 0 else "neutral test residuals"), fontsize=11)
            axes[i, j].set_ylabel("Change / reference eye span")
            axes[i, j].grid(alpha=.15)
            if i == 2:
                axes[i, j].set_xlabel("Estimated relative pitch (degrees; down positive)")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    axes[0, 1].legend(fontsize=8, loc="upper left", ncol=3)
    fig.suptitle("Neutral-trained correction across held-out distances", fontsize=14)
    save(fig, "experiment_01_pose_relationship")

    titles = ["Close / neutral / down\nPreselected example", "Middle / alternating / down\nPreselected example", "Close / alternating / down\nFailure identified after comparison"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 7.9), gridspec_kw={"width_ratios": [1, 1.7]}, constrained_layout=True)
    for i, (rid, title) in enumerate(zip(examples, titles)):
        frame = pred[pred.recording_id == rid]
        for ax in axes[i]:
            ax.set_title(title if ax == axes[i, 0] else "Raw and corrected eyebrow proxy", fontsize=10)
            ax.set_xlabel("Time (seconds)")
            ax.grid(alpha=.15)
            reference = tables[rid]
            for start, end in _runs(reference.reference_frame.to_numpy(), reference.timestamp_s.to_numpy()):
                ax.axvspan(start, end, color="#bfc6ba", alpha=.35)
            for start, end in _runs((~frame.supported_pose).to_numpy(), frame.timestamp_s.to_numpy()):
                ax.axvspan(start, end, facecolor="#dadada", alpha=.35, hatch="//", edgecolor="#b0b0b0", lw=0)
            for interval in annotations[annotations.recording_id == rid].to_dict("records"):
                if interval["uncertain"]:
                    continue
                color = "#b2d9c5" if interval["state"] == "raised" else "#b9cce2"
                ax.axvspan(interval["start_s"], interval["end_s"], color=color, alpha=.24)
        axes[i, 0].plot(frame.timestamp_s, frame.relative_pitch_deg, color="#426b96", lw=1.4)
        axes[i, 0].set_ylabel("Relative pitch (degrees)")
        for method in ["raw", "gam", "mlp"]:
            axes[i, 1].plot(frame.timestamp_s, frame[method + "_height"], color=COLORS[method], lw=1.4, label=method.upper())
        axes[i, 1].set_ylabel("Height / reference eye span")
    axes[0, 1].legend(ncol=3, fontsize=9)
    fig.suptitle("Unsmoothed examples: stability and preservation", fontsize=14)
    fig.supxlabel("Bands: blue = neutral; green = raised; gray = reference; hatching = correction unavailable", fontsize=9)
    save(fig, "experiment_02_trajectories")

    keys = {}
    for row in rows:
        source = "A" if "different head rotations" in row["source_folder"] else "B"
        keys[row["recording_id"]] = row["distance"][0].upper() + "-" + row["direction"][0].upper() + "-" + source
    pd.DataFrame([{"code": keys[r["recording_id"]], "recording_id": r["recording_id"],
                   "original_filename": r["original_filename"], "source_folder": r["source_folder"]} for r in rows]).to_csv(out / "figure_recording_keys.csv", index=False)
    ordered = sorted(rows, key=lambda r: (["close", "middle", "far"].index(r["distance"]), r["direction"], keys[r["recording_id"]][-1]))
    main = metrics[(metrics.scope == "supported_pose") & metrics.method.isin(COLORS)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.0), constrained_layout=True)
    for ax, condition, metric in [(axes[0, 0], "neutral", "rmse"), (axes[0, 1], "raised", "rmse"), (axes[1, 0], "alternating", "auc")]:
        selected = [r for r in ordered if r["eyebrow_condition"] == condition]
        for i, row in enumerate(selected):
            data = main[main.recording_id == row["recording_id"]].set_index("method")
            baseline = data.loc["raw", metric]
            for j, method in enumerate(COLORS):
                x = i + (j - 1) * .16
                val = data.loc[method, metric]
                ax.plot([x, x], [baseline, val], color=COLORS[method], lw=1, alpha=.6)
                ax.scatter(x, val, color=COLORS[method], s=27, marker=["o", "^", "s"][j], label=method.upper() if i == 0 else None)
        ax.set_xticks(range(len(selected)), [keys[r["recording_id"]] for r in selected], rotation=55, ha="right", fontsize=9)
        ax.set_title(f"{condition.title()} (n={len(selected)} recordings)")
        ax.set_ylabel("Reference-relative RMSE" if metric == "rmse" else "Visible-state AUROC")
        ax.set_ylim(bottom=0, top=1.05 if metric == "auc" else None)
        ax.grid(axis="y", alpha=.2)
        if metric == "auc":
            ax.axhline(.5, ls="--", color="#aaaaaa", lw=1)
    axes[0, 0].legend(fontsize=9, ncol=3)
    for d, distance in enumerate(["close", "middle", "far"]):
        subset = main[(main.method == "raw") & (main.distance == distance)]
        offsets = np.linspace(-.18, .18, len(subset))
        for offset, r in zip(offsets, subset.to_dict("records")):
            x = d + offset
            axes[1, 1].plot([x, x], [r["supported_fraction_all_frames"], r["detection_coverage"]], color="#cccccc", lw=1)
        axes[1, 1].scatter(d + offsets, subset.detection_coverage, color=COLORS["raw"], marker="o", s=32, label="Detection" if d == 0 else None)
        axes[1, 1].scatter(d + offsets, subset.supported_fraction_all_frames, color=COLORS["gam"], marker="^", s=32, label="Supported correction" if d == 0 else None)
    axes[1, 1].set_xticks([0, 1, 2], ["Close", "Middle", "Far"])
    axes[1, 1].set_ylim(0, 1.08)
    axes[1, 1].set_ylabel("Fraction of all recording frames")
    axes[1, 1].set_title("Availability (one pair per recording)")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(axis="y", alpha=.2)
    fig.suptitle("Recording-level comparison on matching frames", fontsize=14)
    fig.supxlabel("C/M/F = camera condition; D/U = down/up; A/B = neutral-movement/expression source folder", fontsize=9)
    save(fig, "experiment_03_recording_results")
    print("Saved three experiment figures (PNG and SVG):", out)


def _runs(mask, times):
    """Display-only spans; primary metrics continue to use original frame masks."""
    from .audit import consecutive_runs
    step = float(np.median(np.diff(times))) if len(times) > 1 else .04
    return [(float(times[start]), float(times[end - 1] + step)) for start, end in consecutive_runs(mask)]
