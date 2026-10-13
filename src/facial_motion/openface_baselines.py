"""Global and signer neutral-distance baselines fitted with equal frame weights."""

import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from .openface_data import ROOT, TARGETS, load_neutral_data, sha256


def _signers(values):
    if isinstance(values, str):
        raise ValueError("Supply a sequence of signer identifiers")
    result = list(values)
    if not all(isinstance(value, str) and value.strip() for value in result):
        raise ValueError("Signer identifiers must be nonempty strings")
    return result


class MeanBaseline:
    """Arithmetic means fitted to training rows; prediction accepts signer IDs only."""

    def __init__(self, mode="signer"):
        if mode not in ("global", "signer"):
            raise ValueError("Baseline mode must be 'global' or 'signer'")
        self.mode = mode
        self.global_mean_ = None

    def fit(self, training):
        missing = {"speaker", *TARGETS} - set(training.columns)
        if missing:
            raise ValueError(f"Missing training columns: {sorted(missing)}")
        if training.empty:
            raise ValueError("Cannot fit means to an empty training partition")
        _signers(training.speaker)
        targets = training.loc[:, list(TARGETS)].to_numpy(dtype=float)
        if not np.isfinite(targets).all():
            raise ValueError("Training targets must be finite; no rows are silently dropped")
        self.global_mean_ = targets.mean(axis=0)
        self.signer_means_ = {
            signer: part.loc[:, list(TARGETS)].to_numpy(dtype=float).mean(axis=0)
            for signer, part in training.groupby("speaker", sort=True)
        }
        self.signer_counts_ = {signer: int(n) for signer, n in training.groupby("speaker").size().items()}
        self.n_training_frames_ = len(training)
        return self

    def unknown_signer_mask(self, signers):
        if self.global_mean_ is None:
            raise ValueError("Fit the baseline before prediction")
        return np.array([signer not in self.signer_means_ for signer in _signers(signers)], dtype=bool)

    def predict(self, signers):
        if self.global_mean_ is None:
            raise ValueError("Fit the baseline before prediction")
        signers = _signers(signers)
        if self.mode == "global":
            return np.tile(self.global_mean_, (len(signers), 1))
        if not signers:
            return np.empty((0, len(TARGETS)))
        return np.array([self.signer_means_.get(signer, self.global_mean_) for signer in signers])

    def parameters(self):
        if self.global_mean_ is None:
            raise ValueError("Fit the baseline before reading parameters")
        return {"mode": self.mode, "weighting": "one equal weight per training frame",
                "n_training_frames": self.n_training_frames_, "target_order": list(TARGETS),
                "global_mean": self.global_mean_.tolist(), "signer_counts": self.signer_counts_,
                "signer_means": {signer: mean.tolist() for signer, mean in self.signer_means_.items()},
                "unknown_signer_policy": "global training mean, explicitly flagged; baseline only"}


def score_targets(observed, predicted):
    observed, predicted = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 2 or observed.shape[1] != 2 or not len(observed):
        raise ValueError("Expected equally shaped, nonempty (frames, 2) target/prediction arrays")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Scoring requires finite observations and predictions")
    residual = observed - predicted
    result = {"n_frames": len(observed), "joint_mse": float(np.square(residual).mean())}
    for column, name in enumerate(("inner", "outer")):
        mse = float(np.square(residual[:, column]).mean())
        result.update({f"{name}_mse": mse, f"{name}_rmse": float(np.sqrt(mse)),
                       f"{name}_mae": float(np.abs(residual[:, column]).mean())})
    return result


def hand_example():
    """Unequal recording lengths make accidental recording weighting detectable."""
    training = pd.DataFrame({"speaker": ["A", "A", "A", "A", "B"],
                             "video_name": ["A_short", "A_long", "A_long", "A_long", "B_short"],
                             TARGETS[0]: [2, 6, 6, 6, 10], TARGETS[1]: [4, 8, 8, 8, 14]})
    validation = pd.DataFrame({"speaker": ["A", "B"], TARGETS[0]: [7, 8], TARGETS[1]: [9, 12]})
    models = {name: MeanBaseline(name).fit(training) for name in ("global", "signer")}
    predictions = {name: model.predict(validation.speaker) for name, model in models.items()}
    np.testing.assert_allclose(models["global"].global_mean_, [6, 8.4], rtol=0, atol=1e-12)
    np.testing.assert_allclose(predictions["signer"], [[5, 7], [10, 14]], rtol=0, atol=1e-12)
    scores = {name: score_targets(validation[list(TARGETS)], values) for name, values in predictions.items()}
    np.testing.assert_allclose([scores["global"]["joint_mse"], scores["signer"]["joint_mse"]], [4.58, 4], atol=1e-12)
    return {"illustrative_only": True, "training_rows": training.to_dict("records"),
            "validation_rows": validation.to_dict("records"), "target_order": list(TARGETS),
            "parameters": {name: model.parameters() for name, model in models.items()},
            "predictions": {name: values.tolist() for name, values in predictions.items()},
            "scores": scores, "signer_corrected_measurements": [[2, 2], [-2, -2]],
            "incorrect_recording_weighted_A_mean": [4, 6],
            "unknown_C_fallback": models["signer"].predict(["C"])[0].tolist(),
            "unknown_C_flagged": bool(models["signer"].unknown_signer_mask(["C"])[0])}


def run_baselines(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Baseline output already exists; choose a new directory")
    data = load_neutral_data(root)
    rows = data.rows
    original = {"input_hashes": {}}
    for relative, expected in original["input_hashes"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Frozen pilot input differs: {relative}")
    output.mkdir(parents=True, exist_ok=False)
    rows.to_csv(output / "neutral_rows.csv", index=False)
    rows.iloc[list(dict.fromkeys([0, 1, 2, *rows.index[rows.quality_low_confidence][:3]]))].to_csv(output / "sample_rows.csv", index=False)
    example = hand_example()
    (output / "hand_example.json").write_text(json.dumps(example, indent=2) + "\n")

    metrics, predictions, assignments, parameters, folds, mutation_checks = [], [], [], {}, [], []
    real_example = None
    for fold, (train_index, validation_index) in enumerate(ShuffleSplit(n_splits=4, test_size=0.25, random_state=0).split(rows)):
        training, validation = rows.iloc[train_index], rows.iloc[validation_index]
        if len(training) != 3310 or len(validation) != 1104 or set(train_index) & set(validation_index):
            raise ValueError("Historical partition sizes or row disjointness differ")
        if not set(validation.speaker).issubset(set(training.speaker)):
            raise ValueError("Historical validation includes an unknown signer")
        folds.append({"fold": fold, "training_frames": len(training), "validation_frames": len(validation),
                      "shared_recordings": len(set(training.video_name) & set(validation.video_name)),
                      "training_low_confidence": int(training.quality_low_confidence.sum()),
                      "validation_low_confidence": int(validation.quality_low_confidence.sum())})
        for partition, part in (("training", training), ("validation", validation)):
            keys = part[["neutral_row", "source_row", "video_name", "frame", "speaker", "confidence", "success", "quality_low_confidence"]].copy()
            keys.insert(0, "partition", partition)
            keys.insert(0, "fold", fold)
            assignments.append(keys)
        parameters[str(fold)] = {}
        mutated = rows.copy(deep=True)
        mutated.loc[validation_index, list(TARGETS)] += np.array([100000.0, -200000.0])
        for name in ("global", "signer"):
            model = MeanBaseline(name).fit(training)
            parameters[str(fold)][name] = model.parameters()
            validation_prediction = model.predict(validation.speaker)
            for partition, part in (("training", training), ("validation", validation)):
                score = score_targets(part[list(TARGETS)], model.predict(part.speaker))
                metrics.append(dict(fold=fold, model=name, partition=partition, **score))
            values = validation[["neutral_row", "source_row", "video_name", "frame", "speaker", "confidence", "success", "quality_low_confidence"]].copy()
            values.insert(0, "model", name)
            values.insert(0, "fold", fold)
            values["unknown_signer"] = model.unknown_signer_mask(validation.speaker)
            values["used_global_fallback"] = values.unknown_signer & (name == "signer")
            for column, label in enumerate(("inner", "outer")):
                values[f"observed_{label}"] = validation[TARGETS[column]].to_numpy()
                values[f"predicted_{label}"] = validation_prediction[:, column]
                values[f"corrected_{label}"] = values[f"observed_{label}"] - values[f"predicted_{label}"]
            predictions.append(values)
            refitted = MeanBaseline(name).fit(mutated.iloc[train_index])
            np.testing.assert_array_equal(refitted.global_mean_, model.global_mean_)
            for signer in model.signer_means_:
                np.testing.assert_array_equal(refitted.signer_means_[signer], model.signer_means_[signer])
            np.testing.assert_array_equal(refitted.predict(validation.speaker), validation_prediction)
            original_mse = score_targets(validation[list(TARGETS)], validation_prediction)["joint_mse"]
            mutated_mse = score_targets(mutated.iloc[validation_index][list(TARGETS)], validation_prediction)["joint_mse"]
            if np.isclose(original_mse, mutated_mse):
                raise AssertionError("Held-out mutation did not change the scoring target")
            mutation_checks.append({"fold": fold, "model": name, "max_parameter_change": 0.0,
                                    "max_prediction_change": 0.0, "original_joint_mse": original_mse,
                                    "mutated_target_joint_mse": mutated_mse})
            if real_example is None and name == "signer":
                first = values.iloc[0]
                signer = first.speaker
                real_example = {"fold": fold, "video_name": first.video_name, "frame": int(first.frame),
                                "speaker": signer, "training_frames_for_signer": model.signer_counts_[signer],
                                "neutral_row": int(first.neutral_row), "target_order": list(TARGETS),
                                "observed": [float(first.observed_inner), float(first.observed_outer)],
                                "predicted_neutral": validation_prediction[0].tolist(),
                                "corrected": [float(first.corrected_inner), float(first.corrected_outer)],
                                "confidence": float(first.confidence), "quality_low_confidence": bool(first.quality_low_confidence)}

    metrics = pd.DataFrame(metrics)
    predictions = pd.concat(predictions, ignore_index=True)
    metrics.to_csv(output / "historical_metrics.csv", index=False)
    predictions.to_csv(output / "historical_predictions.csv", index=False)
    pd.concat(assignments, ignore_index=True).to_csv(output / "historical_assignments.csv", index=False)
    score_columns = [column for column in metrics if column.endswith(("_mse", "_mae", "_rmse"))]
    summary = metrics[metrics.partition.eq("validation")].groupby("model")[score_columns].mean().reset_index()
    summary.to_csv(output / "historical_summary.csv", index=False)
    (output / "training_means.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (output / "real_example.json").write_text(json.dumps(real_example, indent=2) + "\n")

    # Re-score files after serialization, independently of the in-memory scores.
    saved = pd.read_csv(output / "historical_predictions.csv")
    for (fold, name), part in saved.groupby(["fold", "model"]):
        scored = score_targets(part[["observed_inner", "observed_outer"]], part[["predicted_inner", "predicted_outer"]])
        expected = metrics[metrics.fold.eq(fold) & metrics.model.eq(name) & metrics.partition.eq("validation")].iloc[0]
        np.testing.assert_allclose([scored[key] for key in score_columns], expected[score_columns].to_numpy(dtype=float), rtol=1e-12, atol=1e-12)
    for relative, expected in {**original["input_hashes"], **data.provenance["input_sha256"]}.items():
        if sha256(root / relative) != expected:
            raise AssertionError(f"Preserved input changed during baseline execution: {relative}")
    verification = {"passed": True, "hand_calculations_passed": True,
                    "held_out_target_mutation_checks": mutation_checks,
                    "saved_validation_metrics_recalculated": True, "openface_inputs_unchanged": True,
                    "quality_rows_excluded": 0, "unknown_validation_signers": 0,
                    "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12}
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    source_files = ["src/facial_motion/openface_data.py", "src/facial_motion/openface_baselines.py",
                    "tests/test_openface_baselines.py", "docs/methodology.md", "docs/methodology.md"]
    metadata = {"run_at_oslo": datetime.now(ZoneInfo("Europe/Oslo")).isoformat(), "phase": 1,
                "scope": "Global/signer means on historical frame splits only", "platform": platform.platform(),
                "python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")},
                "inputs": data.provenance, "folds": folds, "splitter": {"name": "ShuffleSplit", "n_splits": 4, "test_size": 0.25, "random_state": 0},
                "weighting": "equal training-frame weights; equal validation-frame weights within each fold; mean validation metric over four folds",
                "source_sha256": {relative: sha256(root / relative) for relative in source_files},
                "artifact_sha256": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
                "neural_or_ridge_models_trained": False, 
                "interpretation": "Within-recording neutral prediction; no new-recording or expression-preservation claim."}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/openface/baselines")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = run_baselines(args.output, args.root)
    print("PASS: 4,414 frames / 63 recordings / 8 signers; all 65 low-confidence flags retained.")
    print("PASS: frame-weighted hand calculations, held-out-target independence, saved score recalculation and input preservation.")
    print("Historical validation metrics (mean over four folds):")
    print(summary.to_string(index=False))
    print("Saved:", args.output.resolve())



if __name__ == "__main__":
    main()
