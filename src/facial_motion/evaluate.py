"""Evaluate pilot pitch correction with nested validation across camera distances."""

import csv
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .acquire import ROOT, read_manifest, sha256
from .extract import load_config
from .metrics import preservation_auc, stability, state_labels
from .models import CalibrationModel, balanced_samples
from .validate import inner_folds, outer_folds, validate_manifest


def fit_model(method, penalty, rows, tables, config, seed=42):
    if any(r["eyebrow_condition"] != "neutral" for r in rows):
        raise ValueError("Correction fitting is restricted to neutral recordings")
    training = config["training"]
    if training["mlp_hidden_units"] != 8:
        raise ValueError("The committed architecture has eight hidden units")
    samples = balanced_samples([tables[r["recording_id"]] for r in rows], training["max_samples_per_recording"])
    return CalibrationModel(method, penalty, seed, training["gam_df"], training["mlp_max_iter"]).fit(
        samples.relative_pitch_deg, samples.target)


def choose_penalty(method, rows, tables, config):
    trials = []
    for penalty in config["training"][method + "_penalties"]:
        scores, coverages = [], []
        trial = {"method": method, "penalty": penalty, "status": "ok", "warnings": []}
        try:
            for train, validation in inner_folds(rows):
                model = fit_model(method, penalty, train, tables, config)
                trial["warnings"].extend(model.warnings)
                for row in validation:
                    frame = tables[row["recording_id"]]
                    reference = frame.loc[frame.reference_frame & frame.valid_geometry, "inner_brow_height"].median()
                    corrected, support = model.apply(frame.relative_pitch_deg, frame.inner_brow_height)
                    mask = support & frame.valid_geometry.to_numpy() & ~frame.reference_frame.to_numpy()
                    score = stability(corrected, reference, mask)
                    if score["n"] == 0:
                        raise ValueError("No supported non-reference frames in inner validation")
                    scores.append(score["rmse"])
                    coverages.append(float(mask.sum() / max(1, (~frame.reference_frame).sum())))
            trial.update(score=float(np.mean(scores)), mean_supported_nonreference_fraction=float(np.mean(coverages)))
        except (ValueError, np.linalg.LinAlgError, PerfectSeparationError) as error:
            trial.update(status="failed", score=None, error=str(error))
        trials.append(trial)
    valid = [t for t in trials if t["status"] == "ok"]
    if not valid:
        raise ValueError(f"All {method} candidates failed: {trials}")
    # Declared grid order breaks ties; no held-out measurements enter this decision.
    selected = min(valid, key=lambda t: t["score"])["penalty"]
    return selected, trials


def compare_recording(row, frame, models, intervals):
    valid = frame.valid_geometry.to_numpy() & frame.inner_brow_height.notna().to_numpy() & frame.relative_pitch_deg.notna().to_numpy()
    reference_mask = frame.reference_frame.to_numpy() & valid
    if not reference_mask.any():
        raise ValueError(f"No usable reference for {row['recording_id']}")
    reference_height = float(frame.loc[reference_mask, "inner_brow_height"].median())
    values = {"raw": frame.inner_brow_height.to_numpy()}
    supports = []
    for name, model in models.items():
        values[name], support = model.apply(frame.relative_pitch_deg, frame.inner_brow_height)
        supports.append(support & valid)
    common_support = supports[0]
    if any(not np.array_equal(common_support, s) for s in supports[1:]):
        raise ValueError("Methods have different pose support; use identical training samples")
    labels = state_labels(frame.timestamp_s, intervals) if row["eyebrow_condition"] == "alternating" else np.full(len(frame), np.nan)
    scored, predictions = [], frame[["recording_id", "frame_index", "timestamp_s", "relative_pitch_deg", "reference_frame", "valid_geometry"]].copy()
    predictions["supported_pose"] = common_support
    predictions["visible_state"] = labels
    for method, height in values.items():
        predictions[method + "_height"] = height
        for scope, available in [("supported_pose", common_support), ("complete_clip_raw_fallback", valid)]:
            mask = available & ~frame.reference_frame.to_numpy()
            score = {"recording_id": row["recording_id"], "distance": row["distance"], "direction": row["direction"],
                     "condition": row["eyebrow_condition"], "method": method, "scope": scope,
                     "detection_coverage": float(frame.detected.mean()), "valid_geometry_coverage": float(valid.mean()),
                     "supported_fraction_all_frames": float(common_support.mean()),
                     "supported_fraction_valid_frames": float(common_support.sum() / max(1, valid.sum())),
                     "reference_height": reference_height}
            if row["eyebrow_condition"] in {"neutral", "raised"}:
                score.update(stability(height, reference_height, mask))
            else:
                score.update(preservation_auc(height, labels, mask))
                score["annotated_seconds_total"] = sum(float(i["end_s"]) - float(i["start_s"]) for i in intervals if str(i["uncertain"]).lower() != "true")
                score["raised_intervals"] = sum(i["state"] == "raised" and str(i["uncertain"]).lower() != "true" for i in intervals)
            scored.append(score)
    return scored, predictions


def nested_comparison(rows, tables, annotations, config):
    metrics, predictions, decisions = [], [], []
    for held_out, training, test in outer_folds(rows, config["validation"]["distances"]):
        models, fold_trials = {}, []
        for method in ["gam", "mlp"]:
            penalty, trials = choose_penalty(method, training, tables, config)
            fold_trials.extend(trials)
            seeds = config["training"]["mlp_seeds"] if method == "mlp" else [42]
            for seed in seeds:
                name = "gam" if method == "gam" else "mlp" if seed == 42 else f"mlp_seed_{seed}"
                models[name] = fit_model(method, penalty, training, tables, config, seed)
        decisions.append({"held_out_distance": held_out, "neutral_training_ids": [r["recording_id"] for r in training],
                          "test_ids": [r["recording_id"] for r in test], "inner_trials": fold_trials,
                          "selected_models": {name: {"penalty": m.penalty, "seed": m.seed, "pitch_low": m.low, "pitch_high": m.high,
                              "x_scaler_mean": m.x_scaler.mean_.tolist(), "x_scaler_scale": m.x_scaler.scale_.tolist(),
                              "y_scaler_mean": m.y_scaler.mean_.tolist(), "y_scaler_scale": m.y_scaler.scale_.tolist(),
                              "warnings": m.warnings} for name, m in models.items()}})
        for row in test:
            intervals = [a for a in annotations if a["recording_id"] == row["recording_id"]]
            scores, predicted = compare_recording(row, tables[row["recording_id"]], models, intervals)
            for score in scores:
                score["held_out_distance"] = held_out
            metrics.extend(scores)
            predictions.append(predicted)
    metrics = pd.DataFrame(metrics)
    # Paired improvements retain each recording and scope; positive means lower error or higher AUROC.
    keys = ["recording_id", "scope"]
    raw = metrics.loc[metrics.method == "raw", keys + ["rmse", "mae", "auc"]].rename(
        columns={"rmse": "raw_rmse", "mae": "raw_mae", "auc": "raw_auc"})
    metrics = metrics.merge(raw, on=keys, validate="many_to_one")
    metrics["rmse_improvement_from_raw"] = metrics.raw_rmse - metrics.rmse
    metrics["auc_change_from_raw"] = metrics.auc - metrics.raw_auc
    return metrics, pd.concat(predictions, ignore_index=True), decisions


def run_evaluation(root=ROOT, output=None):
    root = Path(root)
    out = Path(output) if output is not None else root / "results/face-landmarker"
    if out.exists():
        raise FileExistsError("Evaluation output already exists; choose a fresh directory")
    config = load_config(root / "config.yaml")
    if config["protocol_status"] != "frozen" or config["orientation"]["sign_verified"] is not True:
        raise ValueError("Protocol is draft. Verify landmarks, pitch signs, references, duplicates and intervals before freezing config.yaml.")
    rows = validate_manifest(read_manifest(root))
    with (root / "data/references.csv").open(newline="") as f:
        references = {r["recording_id"]: r for r in csv.DictReader(f)}
    with (root / "data/annotations.csv").open(newline="") as f:
        annotations = list(csv.DictReader(f))
    tables = {}
    inputs = ["config.yaml", "data/manifest.csv", "data/references.csv", "data/annotations.csv"]
    inputs.extend(p for p in ["docs/methodology.md", "data/qa_frames.csv", "data/audit_review.json",
                              "results/audit/protocol_freeze.json"] if (root / p).exists())
    for row in rows:
        rid = row["recording_id"]
        ref = references.get(rid)
        if ref is None or ref["verified"].lower() != "true":
            raise ValueError(f"Verified raw-video reference missing: {rid}")
        metadata = json.loads((root / f"results/pilot/{rid}_metadata.json").read_text())
        if metadata["video_sha256"] != row["sha256"] or metadata["model_sha256"] != sha256(root / config["model"]["path"]):
            raise ValueError(f"Source or model changed since extraction: {rid}")
        if metadata["config"]["landmarks"] != config["landmarks"] or metadata["config"]["orientation"]["pitch_sign"] != config["orientation"]["pitch_sign"]:
            raise ValueError(f"Feature configuration changed; rederive cached measurements: {rid}")
        if not metadata["reference_applied"] or not np.allclose([metadata["reference_start_s"], metadata["reference_end_s"]], [float(ref["start_s"]), float(ref["end_s"])]):
            raise ValueError(f"Cached measurements do not use the verified reference: {rid}")
        rel = f"data/cache/{rid}_measurements.csv"
        tables[rid] = pd.read_csv(root / rel)
        inputs.append(rel)
        if row["eyebrow_condition"] == "alternating":
            intervals = [a for a in annotations if a["recording_id"] == rid]
            labels = state_labels(tables[rid].timestamp_s, intervals)
            if len(np.unique(labels[np.isfinite(labels)])) != 2:
                raise ValueError(f"Both visible eyebrow states need independent annotation: {rid}")
    metrics, predictions, decisions = nested_comparison(rows, tables, annotations, config)
    out.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(out / "recording_metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    summary = metrics.groupby(["distance", "condition", "method", "scope"], dropna=False)[
        ["rmse", "mae", "auc", "detection_coverage", "supported_fraction_all_frames"]].mean().reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    packages = ["mediapipe", "numpy", "scipy", "pandas", "statsmodels", "scikit-learn", "opencv-contrib-python"]
    metadata = {"run_at_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                "python": platform.python_version(), "config": config, "folds": decisions,
                "input_hashes": {p: sha256(root / p) for p in inputs},
                "source_hashes": {str(p.relative_to(root)): sha256(p) for p in sorted((root / "src/facial_motion").glob("*.py"))},
                "packages": {p: importlib.metadata.version(p) for p in packages},
                "interpretation": "One-participant recording-distance transfer with a per-recording reference. No population p-values."}
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Evaluated {len(rows)} recordings across three distance folds: {out}")
    return metrics, predictions, decisions
