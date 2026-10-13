"""Fixed ridge and MLP evaluation on frame splits of released OpenFace measurements."""

import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

from .openface_baselines import MeanBaseline, score_targets
from .openface_data import ROOT, POSE_COLUMNS, TARGETS, load_neutral_data, sha256


CONFIG_PATH = "configs/openface-reproduction.json"
NUMERIC_FEATURES = ("pose_Rx", "pose_Rx_cos", "pose_Tx", "pose_Ry", "pose_Ry_cos",
                    "pose_Ty", "pose_Rz", "pose_Rz_cos", "pose_Tz")
CATEGORIES = ("speaker", "sentence")
KEYS = ["neutral_row", "source_row", "video_name", "frame", "speaker", "sentence",
        "confidence", "success", "quality_low_confidence"]
FIXED_CONFIG = {
    "phase": 3, "evaluation": "historical_frame_split",
    "splitter": {"n_splits": 4, "test_size": 0.25, "random_state": 0},
    "primary_seed": 42, "sensitivity_seeds": [7, 19], "ridge": {"alpha": 0.001},
    "mlp": {"hidden_layer_sizes": [40], "activation": "relu", "solver": "adam",
            "alpha": 0.0001, "max_iter": 1500, "early_stopping": False},
    "target_scaling": False,
    "input_scaling": "StandardScaler on complete training numeric/one-hot matrix",
    "quality_exclusions": 0, "supplementary_max_iter": 3000,
}


def now():
    return datetime.now(ZoneInfo("Europe/Oslo")).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def numeric_features(rows):
    missing = set(POSE_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"Missing OpenFace input columns: {sorted(missing)}")
    pose = rows.loc[:, list(POSE_COLUMNS)].to_numpy(dtype=float)
    if not len(pose) or not np.isfinite(pose).all():
        raise ValueError("OpenFace inputs must be nonempty and finite; no rows are dropped")
    return np.column_stack([value for i in range(3)
                            for value in (pose[:, i], np.cos(pose[:, i]), pose[:, i + 3])])


class OpenFacePreprocessor:
    """Training-fitted categories and complete design scaling; never reads targets."""

    def _categories(self, rows):
        missing = set(CATEGORIES) - set(rows.columns)
        if missing:
            raise ValueError(f"Missing OpenFace input columns: {sorted(missing)}")
        values = rows.loc[:, list(CATEGORIES)]
        if not values.map(lambda x: isinstance(x, str) and bool(x.strip())).all().all():
            raise ValueError("Signer/sentence identifiers must be nonempty strings")
        return values

    def fit(self, training):
        numeric = numeric_features(training)
        categories = self._categories(training)
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None)
        indicators = self.encoder_.fit_transform(categories)
        self.feature_names_ = [*NUMERIC_FEATURES, *self.encoder_.get_feature_names_out(CATEGORIES)]
        self.scaler_ = StandardScaler().fit(np.column_stack([numeric, indicators]))
        return self

    def unknown_sentence_mask(self, rows):
        if not hasattr(self, "encoder_"):
            raise ValueError("Fit OpenFace preprocessing before transformation")
        return ~self._categories(rows).sentence.isin(self.encoder_.categories_[1]).to_numpy()

    def design(self, rows):
        if not hasattr(self, "encoder_"):
            raise ValueError("Fit OpenFace preprocessing before transformation")
        numeric, categories = numeric_features(rows), self._categories(rows)
        if not categories.speaker.isin(self.encoder_.categories_[0]).all():
            raise ValueError("Unknown signer: personalized OpenFace models require a known signer")
        return np.column_stack([numeric, self.encoder_.transform(categories)])

    def transform(self, rows):
        return self.scaler_.transform(self.design(rows))

    def state(self):
        if not hasattr(self, "encoder_"):
            raise ValueError("Fit OpenFace preprocessing before reading state")
        return {"numeric_feature_order": list(NUMERIC_FEATURES), "angle_units": "released radians",
                "feature_order": list(self.feature_names_), "target_order": list(TARGETS),
                "categories": {name: values.tolist() for name, values in zip(CATEGORIES, self.encoder_.categories_)},
                "scaler_mean": self.scaler_.mean_.tolist(), "scaler_var": self.scaler_.var_.tolist(),
                "scaler_scale": self.scaler_.scale_.tolist(),
                "scaler_n_samples_seen": int(self.scaler_.n_samples_seen_), "target_scaling": False,
                "unknown_signer_policy": "reject", "unknown_sentence_policy": "zero sentence indicators; flag"}


def make_estimator(model, seed=None, max_iter=1500):
    if model == "ridge":
        return Ridge(alpha=0.001)
    if model != "mlp" or seed not in (42, 7, 19) or max_iter not in (1500, 3000):
        raise ValueError("Use only the fixed ridge/MLP settings and seeds")
    return MLPRegressor(hidden_layer_sizes=(40,), activation="relu", solver="adam",
                        alpha=0.0001, max_iter=max_iter, early_stopping=False, random_state=seed)


def audit_preprocessing(preprocessor, training, validation, all_rows):
    # Independently construct the matrix, without calling the adapter's feature methods.
    blocks = []
    for axis in "xyz":
        angle = training[f"pose_R{axis}"].to_numpy(dtype=float)
        blocks.extend([angle, np.cos(angle), training[f"pose_T{axis}"].to_numpy(dtype=float)])
    for category in CATEGORIES:
        classes = sorted(training[category].unique())
        assert preprocessor.state()["categories"][category] == classes
        blocks.extend([training[category].eq(value).to_numpy(dtype=float) for value in classes])
    expected = np.column_stack(blocks)
    np.testing.assert_array_equal(preprocessor.design(training), expected)
    np.testing.assert_allclose(preprocessor.scaler_.mean_, expected.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(preprocessor.scaler_.var_, expected.var(axis=0), rtol=1e-12, atol=1e-12)
    assert preprocessor.scaler_.n_samples_seen_ == len(training)
    assert len(preprocessor.feature_names_) == 27
    # Recreate the inspected helper's column values without executing upstream scripts.
    legacy_blocks = [all_rows[f"pose_R{axis}"].to_numpy(dtype=float) if kind == "angle"
                     else np.cos(all_rows[f"pose_R{axis}"].to_numpy(dtype=float)) if kind == "cos"
                     else all_rows[f"pose_T{axis}"].to_numpy(dtype=float)
                     for axis in "xyz" for kind in ("angle", "cos", "position")]
    label_mapping = {}
    for category in CATEGORIES:
        classes = sorted(all_rows[category].unique())
        first_observed = all_rows[category].unique().tolist()
        label_mapping[category] = dict(zip(first_observed, classes))
        legacy_blocks.extend([all_rows[category].eq(value).to_numpy(dtype=float) for value in classes])
    np.testing.assert_array_equal(preprocessor.design(all_rows), np.column_stack(legacy_blocks))
    unknown_sentences = int(preprocessor.unknown_sentence_mask(validation).sum())
    return {"training_only_mean_var_categories_passed": True, "design_width": expected.shape[1],
            "legacy_numerical_matrix_identical": True, "legacy_label_to_actual_category": label_mapping,
            "unknown_validation_sentences": unknown_sentences, "unknown_validation_signers": 0}


def archived_reference(root):
    base = Path(root) / "research/upstream-review/github/processing"
    logs = json.loads((base / "saved/cross-validation/logs.json").read_text())
    matched = [rec for rec in logs if rec["model"] == "mlp"
               and rec["params"].get("hidden_layer_sizes") == [40] and len(rec["features"]) == 27]
    if len(matched) != 1 or matched[0]["target"] != list(TARGETS):
        raise ValueError("Expected one archived 40-unit/27-input inner-outer MLP record")
    return matched[0]


def fit_estimator(estimator, x, y, fit_id):
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(x, y)
    incomplete = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    state = {"fit_id": fit_id, "resolved_estimator_parameters": estimator.get_params(deep=False),
             "fit_seconds": time.monotonic() - started, "completed_at_oslo": now(),
             "fit_complete": not incomplete,
             "convergence_note": "No convergence warning is the estimator stopping criterion, not proof of a global optimum.",
             "warnings": [{"category": item.category.__name__, "message": str(item.message),
                           "filename": item.filename, "lineno": item.lineno} for item in caught],
             "n_iter": int(estimator.n_iter_) if hasattr(estimator, "n_iter_") and estimator.n_iter_ is not None else None}
    if isinstance(estimator, MLPRegressor):
        state.update({"final_training_loss": float(estimator.loss_),
                      "n_parameters": sum(a.size for a in [*estimator.coefs_, *estimator.intercepts_]),
                      "coefs": [a.tolist() for a in estimator.coefs_],
                      "intercepts": [a.tolist() for a in estimator.intercepts_]})
    else:
        state.update({"coefs": estimator.coef_.tolist(), "intercepts": estimator.intercept_.tolist()})
    print(f"FIT {fit_id}: iterations={state['n_iter']}, complete={state['fit_complete']}, warnings={len(caught)}", flush=True)
    return state


def run_reproduction(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("OpenFace reproduction output already exists; choose a fresh directory")
    config = json.loads((root / CONFIG_PATH).read_text())
    if config != FIXED_CONFIG:
        raise ValueError("Configuration differs from the fixed historical protocol")
    data = load_neutral_data(root)
    rows = data.rows
    frozen = {}
    upstream = json.loads((root / "research/upstream-assets.json").read_text())["assets"]
    preserved = {**frozen, **data.provenance["input_sha256"], **{a["local_path"]: a["sha256"] for a in upstream}}
    for relative, expected in preserved.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Preserved input/source hash differs: {relative}")
    archived = archived_reference(root)
    split_indices = list(ShuffleSplit(**config["splitter"]).split(rows))
    output.mkdir(parents=True, exist_ok=False)
    (output / "fit_states").mkdir()
    source_files = ["src/facial_motion/openface_reproduction.py", "src/facial_motion/openface_data.py",
                    "src/facial_motion/openface_baselines.py", "tests/test_openface_reproduction.py",
                    CONFIG_PATH]
    source_hashes = {relative: sha256(root / relative) for relative in source_files}
    plan = {"recorded_before_fitting_at_oslo": now(), "config": config, "input_provenance": data.provenance,
            "target_order": list(TARGETS), "numeric_feature_order": list(NUMERIC_FEATURES),
            "estimator_defaults": {"ridge": make_estimator("ridge").get_params(),
                                   "mlp_seed42": make_estimator("mlp", 42).get_params()},
            "source_sha256": source_hashes, "thread_policy": "one BLAS thread during fitting/scoring",
            "convergence_policy": "retain incomplete primary; audit input/scaling; record remedy before comparisons; one fresh same-seed 3000-iteration supplementary fit only"}
    write_json(output / "run_plan.json", plan)
    write_json(output / "archived_reference.json", archived)
    rows.to_csv(output / "neutral_rows.csv", index=False)

    metrics, predictions, assignments, runs, curves, folds, remedies = [], [], [], [], [], [], []
    preprocessing, means = {}, {}
    example = None

    def record_predictions(fold, model, seed, limit, fit_id, complete, part, predicted, partition):
        score = score_targets(part[list(TARGETS)], predicted)
        metrics.append(dict(fold=fold, model=model, seed=seed, max_iter=limit, fit_id=fit_id,
                            fit_complete=complete, partition=partition, **score))
        frame = part[KEYS].copy()
        for key, value in dict(fold=fold, model=model, seed=seed, max_iter=limit, fit_id=fit_id,
                               fit_complete=complete, partition=partition).items():
            frame[key] = value
        frame["unknown_sentence"] = False  # checked for all historical folds below
        frame["unknown_signer"] = False
        for col, label in enumerate(("inner", "outer")):
            frame[f"observed_{label}"] = part[TARGETS[col]].to_numpy()
            frame[f"predicted_{label}"] = predicted[:, col]
            frame[f"corrected_{label}"] = frame[f"observed_{label}"] - frame[f"predicted_{label}"]
        predictions.append(frame)

    with threadpool_limits(limits=1):
        pools = [{**item, "filepath": Path(item["filepath"]).name} for item in threadpool_info()]
        for fold, (train_ids, val_ids) in enumerate(split_indices):
            training, validation = rows.iloc[train_ids], rows.iloc[val_ids]
            assert (len(training), len(validation)) == (3310, 1104)
            assert set(train_ids).isdisjoint(val_ids)
            assert set(train_ids) | set(val_ids) == set(range(len(rows)))
            assert set(validation.speaker).issubset(set(training.speaker))
            assert int(training.quality_low_confidence.sum() + validation.quality_low_confidence.sum()) == 65
            prep = OpenFacePreprocessor().fit(training)
            audit = audit_preprocessing(prep, training, validation, rows)
            assert audit["unknown_validation_sentences"] == 0
            preprocessing[str(fold)] = {**prep.state(), "audit": audit}
            xtrain, xval = prep.transform(training), prep.transform(validation)
            ytrain = training.loc[:, list(TARGETS)].to_numpy(dtype=float)
            fold_info = {"fold": fold, "training_frames": len(training), "validation_frames": len(validation),
                         "shared_recordings": len(set(training.video_name) & set(validation.video_name)),
                         "training_low_confidence": int(training.quality_low_confidence.sum()),
                         "validation_low_confidence": int(validation.quality_low_confidence.sum())}
            for partition, part in (("training", training), ("validation", validation)):
                keys = part[KEYS].copy()
                keys.insert(0, "partition", partition)
                keys.insert(0, "fold", fold)
                assignments.append(keys)
                fold_info[f"{partition}_joint_target_mean"] = float(part[list(TARGETS)].to_numpy().mean())
                fold_info[f"{partition}_mean_target_sample_std"] = float(part[list(TARGETS)].std().mean())
            folds.append(fold_info)
            means[str(fold)] = {}
            for name in ("global", "signer"):
                model = MeanBaseline(name).fit(training)
                means[str(fold)][name] = model.parameters()
                for partition, part in (("training", training), ("validation", validation)):
                    record_predictions(fold, name, None, None, f"fold{fold}_{name}", True,
                                       part, model.predict(part.speaker), partition)
            for model_name, seed in [("ridge", None), ("mlp", 42), ("mlp", 7), ("mlp", 19)]:
                limit = 1500 if model_name == "mlp" else None
                fit_id = f"fold{fold}_{model_name}" + (f"_seed{seed}_iter{limit}" if seed is not None else "")
                estimator = make_estimator(model_name, seed)
                state = fit_estimator(estimator, xtrain, ytrain, fit_id)
                attempts = [(estimator, state, model_name, limit)]
                if not state["fit_complete"]:
                    # This is recorded before scoring either this fit or its supplementary remedy.
                    remedy = {"recorded_at_oslo": now(), "fold": fold, "seed": seed,
                              "incomplete_fit": fit_id, "trigger": state["warnings"],
                              "input_and_scaling_audit_passed": True, "comparison_used_to_choose_remedy": False,
                              "change": "fresh same-seed supplementary fit; max_iter 1500 to 3000 only"}
                    remedies.append(remedy)
                    write_json(output / "convergence_remedies.json", remedies)
                    supplementary = make_estimator("mlp", seed, max_iter=3000)
                    supplementary_id = f"fold{fold}_mlp_seed{seed}_iter3000"
                    supplementary_state = fit_estimator(supplementary, xtrain, ytrain, supplementary_id)
                    attempts.append((supplementary, supplementary_state, "mlp_supplementary", 3000))
                for fitted, fit_state, label, iterations in attempts:
                    fit_state.update(fold=fold, model=label, seed=seed, max_iter=iterations,
                                     training_frames=len(training), target_order=list(TARGETS))
                    write_json(output / "fit_states" / f"{fit_state['fit_id']}.json", fit_state)
                    runs.append({k: v for k, v in fit_state.items() if k not in ("coefs", "intercepts")})
                    if hasattr(fitted, "loss_curve_"):
                        curves.extend(dict(fit_id=fit_state["fit_id"], fold=fold, seed=seed, max_iter=iterations,
                                           iteration=i + 1, training_loss=float(loss))
                                      for i, loss in enumerate(fitted.loss_curve_))
                    for partition, part, matrix in (("training", training, xtrain), ("validation", validation, xval)):
                        predicted = fitted.predict(matrix)
                        record_predictions(fold, label, seed, iterations, fit_state["fit_id"],
                                           fit_state["fit_complete"], part, predicted, partition)
                        if example is None and label == "mlp" and seed == 42 and partition == "validation":
                            example = {"fold": fold, "fit_id": fit_state["fit_id"],
                                       "row_identity": part.iloc[0][KEYS].to_dict(),
                                       "feature_order": list(prep.feature_names_),
                                       "unscaled_design": prep.design(part.iloc[:1])[0].tolist(),
                                       "scaled_design": matrix[0].tolist(), "target_order": list(TARGETS),
                                       "observed": part.iloc[0][list(TARGETS)].to_numpy(dtype=float).tolist(),
                                       "predicted_neutral": predicted[0].tolist(),
                                       "corrected": (part.iloc[0][list(TARGETS)].to_numpy(dtype=float) - predicted[0]).tolist()}

    metric_table = pd.DataFrame(metrics)
    metric_table.to_csv(output / "historical_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "historical_predictions.csv", index=False)
    pd.concat(assignments, ignore_index=True).to_csv(output / "historical_assignments.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "loss_curves.csv", index=False)
    write_json(output / "preprocessing.json", preprocessing)
    write_json(output / "training_means.json", means)
    write_json(output / "model_runs.json", runs)
    write_json(output / "real_example.json", example)
    if not remedies:
        write_json(output / "convergence_remedies.json", [])
    score_columns = [c for c in metric_table if c.endswith(("_mse", "_rmse", "_mae"))]
    summaries = []
    for (model, seed, limit), part in metric_table.groupby(["model", "seed", "max_iter"], dropna=False, sort=False):
        for partition, group in part.groupby("partition", sort=False):
            summaries.append({"model": model, "seed": seed, "max_iter": limit, "partition": partition,
                              "n_folds": len(group), "all_fits_complete": bool(group.fit_complete.all()),
                              "eligible_four_fold_comparison": bool(group.fit_complete.all() and len(group) == 4),
                              **group[score_columns].mean().to_dict()})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "historical_summary.csv", index=False)

    saved = pd.read_csv(output / "historical_predictions.csv")
    for (fit_id, partition), part in saved.groupby(["fit_id", "partition"]):
        # Independent arithmetic on serialized predictions (no production scorer).
        residual = part[["observed_inner", "observed_outer"]].to_numpy() - part[["predicted_inner", "predicted_outer"]].to_numpy()
        expected = metric_table[metric_table.fit_id.eq(fit_id) & metric_table.partition.eq(partition)].iloc[0]
        np.testing.assert_allclose(np.square(residual).mean(), expected.joint_mse, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(residual, part[["corrected_inner", "corrected_outer"]], rtol=1e-12, atol=1e-12)
        for col, label in enumerate(("inner", "outer")):
            np.testing.assert_allclose([np.square(residual[:, col]).mean(), np.sqrt(np.square(residual[:, col]).mean()), np.abs(residual[:, col]).mean()],
                                       expected[[f"{label}_mse", f"{label}_rmse", f"{label}_mae"]].to_numpy(dtype=float), rtol=1e-12, atol=1e-12)
    accepted = pd.read_csv(root / "results/openface/baselines/historical_metrics.csv")
    baseline = metric_table[metric_table.model.isin(["global", "signer"])]
    pd.testing.assert_frame_equal(baseline[accepted.columns].reset_index(drop=True), accepted,
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    for relative, expected in {**preserved, **source_hashes}.items():
        if sha256(root / relative) != expected:
            raise AssertionError(f"Input/source changed during training: {relative}")
    archive_mse = float(archived["val"]["mean_score"]["mse"])
    comparisons = []
    for rec in summary[summary.model.eq("mlp") & summary.partition.eq("validation")].to_dict("records"):
        comparisons.append({"seed": int(rec["seed"]), "joint_mse": rec["joint_mse"],
                            "all_fits_complete": rec["all_fits_complete"],
                            "difference_from_archive": rec["joint_mse"] - archive_mse,
                            "relative_difference_percent": 100 * (rec["joint_mse"] / archive_mse - 1)})
    diagnostics = []
    for fold in folds:
        for partition, archive_partition in (("training", "train"), ("validation", "val")):
            logged = archived[archive_partition]["folds"][fold["fold"]]
            archived_mean = float(np.mean(logged["mean_true"]))
            diagnostics.append({"fold": fold["fold"], "partition": partition,
                                "current_joint_target_mean": fold[f"{partition}_joint_target_mean"],
                                "archived_joint_target_mean": archived_mean,
                                "archived_per_target_mean": logged["mean_true"],
                                "target_mean_difference": fold[f"{partition}_joint_target_mean"] - archived_mean,
                                "current_mean_target_sample_std": fold[f"{partition}_mean_target_sample_std"],
                                "archived_mean_target_sample_std": float(np.mean(logged["std_true"]))})
    write_json(output / "reproduction_comparison.json", {
        "archived_joint_validation_mse": archive_mse, "primary_seed": 42, "fixed_seed_results": comparisons,
        "archived_seed": "unrecorded", "historical_package_versions": "unrecorded in inspected log",
        "alpha": "explicit 0.0001; omitted in archived CV record; later deployment 0.1 not used",
        "numerical_design_matches_inspected_helper": True,
        "category_difference": "actual sorted category names, training-fitted; legacy labels use first-observed names on sorted class values",
        "target_distribution_diagnostics": diagnostics,
        "scope": "procedural reproduction on pinned released data, not a promise of the exact historical random state or run inputs",
        "no_seed_selection_or_tuning_for_archive_agreement": True})
    write_json(output / "verification.json", {
        "passed": True, "retained_neutral_rows": len(rows), "retained_low_confidence_flags": 65,
        "quality_exclusions": 0, "training_only_scalers_and_categories_checked": 4,
        "numeric_matrix_independently_recalculated": True, "legacy_numerical_matrix_identical": True,
        "all_saved_training_validation_metrics_recalculated": True,
        "baseline_metrics_reproduced": True, "input_source_hashes_unchanged": True,
        "frozen_pilot_hashes_checked": len(frozen), "upstream_assets_checked": len(upstream),
        "prediction_groups_checked": int(saved.groupby(["fit_id", "partition"]).ngroups),
        "primary_mlp_complete_fits": sum(r["fit_complete"] for r in runs if r["model"] == "mlp"),
        "primary_mlp_incomplete_fits": sum(not r["fit_complete"] for r in runs if r["model"] == "mlp"),
        "supplementary_fits": len(remedies), "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12})
    write_json(output / "run_metadata.json", {
        "completed_at_oslo": now(), "phase": 3, "config": config, "folds": folds,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")},
        "threadpools_during_fit": pools, "inputs": data.provenance, "source_sha256": source_hashes,
        "artifact_sha256": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "weighting": "equal frames within each fold; arithmetic mean across four folds; original target units",
        
        "inference_bundle_verified": False,
        "interpretation": "Within-recording neutral prediction only; not new-recording or expression-preservation evidence."})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/openface/frame-split")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = run_reproduction(args.output, args.root)
    print("PASS: retained rows/flags, training-only preprocessing, saved scoring, accepted baselines and input hashes.")
    print(summary.to_string(index=False))
    print("Saved:", args.output.resolve())



if __name__ == "__main__":
    main()
