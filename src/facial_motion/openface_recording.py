"""Fixed evaluation on whole OpenFace recordings of known signers."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

from .openface_baselines import MeanBaseline, score_targets
from .openface_data import ROOT, TARGETS, load_neutral_data, sha256
from .openface_reproduction import (CATEGORIES, KEYS, NUMERIC_FEATURES,
                                   OpenFacePreprocessor, fit_estimator,
                                   make_estimator, now, write_json)


CONFIG_PATH = "configs/openface-recording-evaluation.json"
FOLD_PATH = "configs/recording-folds.csv"
FOLD_SHA256 = "f888c92b5c2a96fc6cc74ebddbdf01895bf00f3197f37037cd91954df58afed0"
FIXED_CONFIG = {
    "phase": 4, "evaluation": "whole_recordings_known_signers",
    "fold_path": FOLD_PATH, "fold_sha256": FOLD_SHA256,
    "primary_seed": 42, "sensitivity_seeds": [7, 19],
    "ridge": {"alpha": .001},
    "identifier_ridge": {"alpha": .001, "inputs": ["speaker", "sentence"]},
    "mlp": {"hidden_layer_sizes": [40], "activation": "relu", "solver": "adam",
            "alpha": .0001, "max_iter": 1500, "early_stopping": False},
    "primary_metric": "mean recording joint MSE across all held-out recordings",
    "training_weighting": "equal frames", "target_scaling": False,
    "input_scaling": "StandardScaler on complete training numeric/one-hot matrix",
    "quality_exclusions": 0, "unknown_signer_policy": "reject",
    "unknown_sentence_policy": "zero sentence indicators before scaling; flag",
    "supplementary_max_iter": 3000,
}
SCORE_COLUMNS = ["joint_mse", "inner_mse", "inner_rmse", "inner_mae",
                 "outer_mse", "outer_rmse", "outer_mae"]
GROUP = ["model", "seed", "max_iter"]


class IdentifierPreprocessor(OpenFacePreprocessor):
    """Only training-fitted signer/sentence indicators and their scaling."""

    def fit(self, training):
        categories = self._categories(training)
        if training.empty:
            raise ValueError("Cannot fit identifier preprocessing to empty training rows")
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None)
        indicators = self.encoder_.fit_transform(categories)
        self.feature_names_ = list(self.encoder_.get_feature_names_out(CATEGORIES))
        self.scaler_ = StandardScaler().fit(indicators)
        return self

    def design(self, rows):
        if not hasattr(self, "encoder_"):
            raise ValueError("Fit identifier preprocessing before transformation")
        categories = self._categories(rows)
        if not categories.speaker.isin(self.encoder_.categories_[0]).all():
            raise ValueError("Unknown signer: personalized OpenFace models require a known signer")
        return self.encoder_.transform(categories)

    def state(self):
        state = super().state()
        state["numeric_feature_order"] = []
        state["angle_units"] = "not applicable; no numeric pose inputs"
        return state


def recording_splits(rows, allocation):
    """Validate one assignment per source recording, then preserve source row order."""
    required = {"video_name", "speaker", "validation_fold"}
    if not required.issubset(allocation.columns) or allocation.empty:
        raise ValueError("Recording allocation must contain video_name, speaker and validation_fold")
    if allocation.video_name.duplicated().any():
        raise ValueError("Each recording must have exactly one validation fold")
    if set(allocation.video_name) != set(rows.video_name):
        raise ValueError("Recording allocation must cover exactly the neutral recordings")
    if not allocation.validation_fold.isin(range(4)).all() or set(allocation.validation_fold) != set(range(4)):
        raise ValueError("Recording folds must be integers 0 through 3, all represented")
    identities = rows.groupby("video_name").speaker
    if identities.nunique().ne(1).any():
        raise ValueError("Each source recording must have one signer")
    expected = allocation.video_name.map(identities.first())
    if not expected.eq(allocation.speaker).all():
        raise ValueError("Allocation signer identities differ from source recordings")
    result = []
    for fold in range(4):
        held = allocation.loc[allocation.validation_fold.eq(fold), "video_name"]
        mask = rows.video_name.isin(held).to_numpy()
        train, validation = np.flatnonzero(~mask), np.flatnonzero(mask)
        if not len(train) or not len(validation):
            raise ValueError("Empty training or validation recording partition")
        if not set(rows.iloc[validation].speaker).issubset(set(rows.iloc[train].speaker)):
            raise ValueError("Validation signer absent from training")
        assert set(rows.iloc[train].video_name).isdisjoint(rows.iloc[validation].video_name)
        result.append((train, validation))
    np.testing.assert_array_equal(np.sort(np.concatenate([v for _, v in result])), np.arange(len(rows)))
    return result


def audit_preprocessor(prep, training, validation, numeric=True):
    """Independent feature/category/scaler arithmetic, including target-free transforms."""
    state = prep.state()

    def design(part):
        blocks = []
        if numeric:
            for axis in "xyz":
                angle = part[f"pose_R{axis}"].to_numpy(dtype=float)
                blocks.extend([angle, np.cos(angle), part[f"pose_T{axis}"].to_numpy(dtype=float)])
        for category in CATEGORIES:
            classes = sorted(training[category].unique())
            assert state["categories"][category] == classes
            blocks.extend([part[category].eq(value).to_numpy(dtype=float) for value in classes])
        return np.column_stack(blocks)

    matrix = design(training)
    np.testing.assert_array_equal(prep.design(training), matrix)
    np.testing.assert_array_equal(prep.design(validation), design(validation))
    np.testing.assert_allclose(state["scaler_mean"], matrix.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state["scaler_var"], matrix.var(axis=0), rtol=1e-12, atol=1e-12)
    assert state["scaler_n_samples_seen"] == len(training)
    target_free = validation.drop(columns=list(TARGETS))
    np.testing.assert_array_equal(prep.transform(validation), prep.transform(target_free))
    np.testing.assert_array_equal(prep.transform(validation)[::-1], prep.transform(target_free.iloc[::-1]))
    return {"passed": True, "training_only_categories_mean_var": True,
            "target_free_and_row_reordering_checked": True, "design_width": matrix.shape[1],
            "unknown_validation_sentences": int(prep.unknown_sentence_mask(validation).sum()),
            "unknown_validation_signers": 0}


def summarize_recordings(recording_metrics):
    """Pool validation recordings equally; frame score weights by retained frame counts."""
    records = []
    held = recording_metrics[recording_metrics.partition.eq("validation")]
    for (model, seed, limit), group in held.groupby(GROUP, dropna=False, sort=False):
        if group.video_name.duplicated().any():
            raise ValueError("A recording has multiple validation scores for the same model/seed/attempt")
        records.append({"model": model, "seed": seed, "max_iter": limit,
                        "n_recordings": len(group), "n_frames": int(group.n_frames.sum()),
                        "n_folds": group.fold.nunique(), "all_fits_complete": bool(group.fit_complete.all()),
                        "eligible_four_fold_comparison": bool(group.fit_complete.all() and group.fold.nunique() == 4),
                        **{f"equal_recording_{c}": float(group[c].mean()) for c in SCORE_COLUMNS},
                        "frame_weighted_joint_mse": float(np.average(group.joint_mse, weights=group.n_frames)),
                        "unknown_sentence_recordings": int(group.unknown_sentence_frames.gt(0).sum()),
                        "unknown_sentence_frames": int(group.unknown_sentence_frames.sum())})
    return pd.DataFrame(records)


def paired_recordings(recording_metrics):
    held = recording_metrics[recording_metrics.partition.eq("validation")]
    pairs = []
    comparisons = [("mlp", seed, control) for seed in (42, 7, 19)
                   for control in ("global", "signer", "ridge", "identifier_ridge")]
    comparisons.append(("ridge", None, "identifier_ridge"))
    for candidate, seed, control in comparisons:
        left = held[held.model.eq(candidate)]
        if candidate == "mlp":
            left = left[left.seed.eq(seed)]
        right = held[held.model.eq(control)]
        for limit, group in left.groupby("max_iter", dropna=False, sort=False):
            joined = group.merge(right, on=["fold", "video_name", "speaker", "sentence"],
                                 suffixes=("_candidate", "_control"), validate="one_to_one")
            if len(joined) != len(group):
                raise ValueError("Paired models must score identical recordings")
            np.testing.assert_array_equal(joined.n_frames_candidate, joined.n_frames_control)
            for r in joined.to_dict("records"):
                pairs.append({"candidate": candidate, "seed": seed, "max_iter": limit, "control": control,
                              **{k: r[k] for k in ("fold", "video_name", "speaker", "sentence")},
                              "n_frames": r["n_frames_candidate"],
                              "fit_complete": r["fit_complete_candidate"] and r["fit_complete_control"],
                              "candidate_joint_mse": r["joint_mse_candidate"],
                              "control_joint_mse": r["joint_mse_control"],
                              "joint_mse_difference": r["joint_mse_candidate"] - r["joint_mse_control"],
                              "candidate_improves": r["joint_mse_candidate"] < r["joint_mse_control"],
                              "unknown_sentence_frames": r["unknown_sentence_frames_candidate"]})
    return pd.DataFrame(pairs)


def run_recording_evaluation(output, root=ROOT, save_bundles=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("OpenFace recording output already exists; choose a fresh directory")
    config = json.loads((root / CONFIG_PATH).read_text())
    if config != FIXED_CONFIG:
        raise ValueError("Configuration differs from the fixed recording protocol")
    if sha256(root / FOLD_PATH) != FOLD_SHA256:
        raise ValueError("Fixed recording allocation hash differs")
    data = load_neutral_data(root)
    rows = data.rows
    allocation = pd.read_csv(root / FOLD_PATH)
    splits = recording_splits(rows, allocation)
    frozen = {}
    upstream = json.loads((root / "research/upstream-assets.json").read_text())["assets"]
    preserved = {**frozen, **data.provenance["input_sha256"], FOLD_PATH: FOLD_SHA256,
                 **{a["local_path"]: a["sha256"] for a in upstream}}
    for relative, expected in preserved.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Preserved input/source hash differs: {relative}")
    source_files = ["src/facial_motion/openface_recording.py", "src/facial_motion/openface_reproduction.py",
                    "src/facial_motion/openface_baselines.py", "src/facial_motion/openface_data.py",
                    "tests/test_openface_recording.py", CONFIG_PATH, "docs/methodology.md",
                    "docs/methodology.md", "docs/methodology.md"]
    if save_bundles:
        from .openface_bundle import save_bundle
        source_files.extend(["src/facial_motion/openface_bundle.py", "docs/methodology.md",
                             "docs/methodology.md"])
    source_hashes = {p: sha256(root / p) for p in source_files}
    output.mkdir(parents=True, exist_ok=False)
    (output / "fit_states").mkdir()
    write_json(output / "run_plan.json", {
        "recorded_before_fitting_at_oslo": now(), "config": config, "input_provenance": data.provenance,
        "source_sha256": source_hashes, "target_order": list(TARGETS),
        "numeric_feature_order": list(NUMERIC_FEATURES),
        "estimator_defaults": {"ridge": make_estimator("ridge").get_params(),
                               "mlp_seed42": make_estimator("mlp", 42).get_params()},
        "thread_policy": "one BLAS thread during fitting/scoring",
        "convergence_policy": "retain incomplete primary; audit inputs/scaling; record one fresh same-seed 3000-iteration supplementary remedy before scoring"})
    rows.to_csv(output / "neutral_rows.csv", index=False)
    allocation.to_csv(output / "recording_allocation.csv", index=False)
    assignments, predictions, metrics, recording_metrics, runs, curves = [], [], [], [], [], []
    preprocessing, means, folds, remedies, coverage = {}, {}, [], [], []
    bundles = []

    def persist(prep, estimator, name, fold, fit_id, seed=None, limit=None):
        if not save_bundles:
            return None
        path = output / "bundles" / f"{fit_id}.json"
        bundle = save_bundle(path, prep, estimator, name, {
            "fit_id": fit_id, "fold": fold, "seed": seed, "max_iter": limit,
            "config": config, "input_provenance": data.provenance,
            "source_sha256": source_hashes, "recording_allocation_sha256": FOLD_SHA256,
            "training_identity_sha256": sha256(output / "neutral_rows.csv"),
            "training_rows": rows.iloc[splits[fold][0]].neutral_row.tolist(),
            "validation_rows": rows.iloc[splits[fold][1]].neutral_row.tolist(),
            "training_identity_note": "neutral_rows.csv hash plus exact training/validation row IDs"})
        bundles.append({"fit_id": fit_id, "fold": fold, "model": name, "seed": seed,
                        "max_iter": limit, "path": str(path.relative_to(output)), "sha256": sha256(path)})
        return bundle

    def check_bundle(bundle, part, predicted):
        if bundle is not None:
            result = bundle.predict(part.drop(columns=list(TARGETS)))
            np.testing.assert_allclose(result[["predicted_inner", "predicted_outer"]], predicted,
                                       rtol=1e-12, atol=1e-12)

    def record(fold, model, seed, limit, fit_id, complete, part, predicted, partition, unknown):
        info = dict(fold=fold, model=model, seed=seed, max_iter=limit, fit_id=fit_id,
                    fit_complete=complete, partition=partition)
        metrics.append({**info, **score_targets(part[list(TARGETS)], predicted)})
        frame = part[KEYS].copy()
        for key, value in info.items():
            frame[key] = value
        frame["unknown_sentence"], frame["unknown_signer"] = unknown, False
        for i, label in enumerate(("inner", "outer")):
            frame[f"observed_{label}"] = part[TARGETS[i]].to_numpy()
            frame[f"predicted_{label}"] = predicted[:, i]
            frame[f"corrected_{label}"] = frame[f"observed_{label}"] - frame[f"predicted_{label}"]
        predictions.append(frame)
        for video, group in frame.groupby("video_name", sort=False):
            recording_metrics.append({**info, "video_name": video, "speaker": group.speaker.iloc[0],
                "sentence": group.sentence.iloc[0], "unknown_sentence_frames": int(group.unknown_sentence.sum()),
                "low_confidence_frames": int(group.quality_low_confidence.sum()),
                **score_targets(group[["observed_inner", "observed_outer"]], group[["predicted_inner", "predicted_outer"]])})

    with threadpool_limits(limits=1):
        pools = [{**item, "filepath": Path(item["filepath"]).name} for item in threadpool_info()]
        for fold, (train_ids, val_ids) in enumerate(splits):
            training, validation = rows.iloc[train_ids], rows.iloc[val_ids]
            full = OpenFacePreprocessor().fit(training)
            identifiers = IdentifierPreprocessor().fit(training)
            audits = {"full": audit_preprocessor(full, training, validation),
                      "identifiers": audit_preprocessor(identifiers, training, validation, numeric=False)}
            preprocessing[str(fold)] = {"full": {**full.state(), "audit": audits["full"]},
                                        "identifiers": {**identifiers.state(), "audit": audits["identifiers"]}}
            ytrain = training.loc[:, list(TARGETS)].to_numpy(dtype=float)
            matrices = {"full": (full.transform(training), full.transform(validation)),
                        "identifiers": (identifiers.transform(training), identifiers.transform(validation))}
            unknown = {name: full.unknown_sentence_mask(part)
                       for name, part in (("training", training), ("validation", validation))}
            np.testing.assert_array_equal(unknown["validation"], identifiers.unknown_sentence_mask(validation))
            fold_info = {"fold": fold, "shared_recordings": 0, "unknown_validation_signers": 0}
            for partition, part in (("training", training), ("validation", validation)):
                assigned = part[KEYS].copy()
                assigned.insert(0, "partition", partition)
                assigned.insert(0, "fold", fold)
                assigned["unknown_sentence"] = unknown[partition]
                assignments.append(assigned)
                fold_info.update({f"{partition}_frames": len(part), f"{partition}_recordings": part.video_name.nunique(),
                                  f"{partition}_low_confidence": int(part.quality_low_confidence.sum())})
                unseen = part.loc[unknown[partition]]
                coverage.append({"fold": fold, "partition": partition, "n_frames": len(part),
                    "n_recordings": part.video_name.nunique(), "unknown_sentence_frames": len(unseen),
                    "unknown_sentence_recordings": unseen.video_name.nunique(),
                    "unknown_sentence_frame_fraction": len(unseen) / len(part),
                    "unknown_sentence_recording_fraction": unseen.video_name.nunique() / part.video_name.nunique(),
                    "unknown_sentences": sorted(unseen.sentence.unique())})
            assert fold_info["training_low_confidence"] + fold_info["validation_low_confidence"] == 65
            folds.append(fold_info)
            means[str(fold)] = {}
            for name in ("global", "signer"):
                baseline = MeanBaseline(name).fit(training)
                means[str(fold)][name] = baseline.parameters()
                bundle = persist(full, baseline, name, fold, f"fold{fold}_{name}")
                for partition, part in (("training", training), ("validation", validation)):
                    predicted = baseline.predict(part.speaker)
                    check_bundle(bundle, part, predicted)
                    record(fold, name, None, None, f"fold{fold}_{name}", True, part,
                           predicted, partition, unknown[partition])
            for name, seed in [("ridge", None), ("identifier_ridge", None), ("mlp", 42), ("mlp", 7), ("mlp", 19)]:
                design_key = "identifiers" if name == "identifier_ridge" else "full"
                xtrain, xval = matrices[design_key]
                limit = 1500 if name == "mlp" else None
                fit_id = f"fold{fold}_{name}" + (f"_seed{seed}_iter{limit}" if seed is not None else "")
                estimator = make_estimator("ridge" if name == "identifier_ridge" else name, seed)
                state = fit_estimator(estimator, xtrain, ytrain, fit_id)
                attempts = [(estimator, state, limit)]
                if not state["fit_complete"]:
                    remedy = {"recorded_at_oslo": now(), "fold": fold, "seed": seed,
                              "incomplete_fit": fit_id, "trigger": state["warnings"],
                              "input_and_scaling_audit_passed": True, "comparison_used_to_choose_remedy": False,
                              "change": "fresh same-seed supplementary fit; max_iter 1500 to 3000 only"}
                    remedies.append(remedy)
                    write_json(output / "convergence_remedies.json", remedies)
                    supplemental = make_estimator("mlp", seed, 3000)
                    supplemental_id = f"fold{fold}_mlp_seed{seed}_iter3000"
                    attempts.append((supplemental, fit_estimator(supplemental, xtrain, ytrain, supplemental_id), 3000))
                for fitted, saved_state, attempt_limit in attempts:
                    saved_state.update({"model": name, "seed": seed, "max_iter": attempt_limit,
                                        "fold": fold, "design": design_key})
                    write_json(output / "fit_states" / (saved_state["fit_id"] + ".json"), saved_state)
                    runs.append(saved_state)
                    bundle = persist(identifiers if design_key == "identifiers" else full,
                                     fitted, name, fold, saved_state["fit_id"], seed, attempt_limit)
                    if name == "mlp":
                        curves.extend({"fit_id": saved_state["fit_id"], "iteration": i + 1, "training_loss": float(loss)}
                                      for i, loss in enumerate(fitted.loss_curve_))
                    # Predict using prepared inputs only, with no validation target values.
                    for partition, part, matrix in (("training", training, xtrain), ("validation", validation, xval)):
                        predicted = fitted.predict(matrix)
                        check_bundle(bundle, part, predicted)
                        record(fold, name, seed, attempt_limit, saved_state["fit_id"], saved_state["fit_complete"],
                               part, predicted, partition, unknown[partition])

    metric_table, recording_table = pd.DataFrame(metrics), pd.DataFrame(recording_metrics)
    prediction_table = pd.concat(predictions, ignore_index=True)
    assignment_table = pd.concat(assignments, ignore_index=True)
    summary = summarize_recordings(recording_table)
    pairs = paired_recordings(recording_table)
    signer_rows = []
    for (model, seed, limit, signer), group in recording_table[recording_table.partition.eq("validation")].groupby([*GROUP, "speaker"], dropna=False, sort=False):
        signer_rows.append({"model": model, "seed": seed, "max_iter": limit, "speaker": signer,
                           "n_recordings": len(group), "n_frames": int(group.n_frames.sum()),
                           "all_fits_complete": bool(group.fit_complete.all()),
                           **{f"equal_recording_{c}": float(group[c].mean()) for c in SCORE_COLUMNS},
                           "frame_weighted_joint_mse": float(np.average(group.joint_mse, weights=group.n_frames))})
    for filename, table in [("fold_metrics.csv", metric_table), ("recording_metrics.csv", recording_table),
                            ("predictions.csv", prediction_table), ("assignments.csv", assignment_table),
                            ("summary.csv", summary), ("signer_metrics.csv", pd.DataFrame(signer_rows)),
                            ("paired_recording_differences.csv", pairs), ("loss_curves.csv", pd.DataFrame(curves))]:
        table.to_csv(output / filename, index=False)
    write_json(output / "preprocessing.json", preprocessing)
    write_json(output / "training_means.json", means)
    write_json(output / "model_runs.json", runs)
    write_json(output / "unknown_sentence_coverage.json", coverage)
    write_json(output / "convergence_remedies.json", remedies)
    if save_bundles:
        write_json(output / "bundle_manifest.json", {"schema_version": 1, "bundles": bundles,
            "same_process_predictions_checked": True, "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12})
    saved = pd.read_csv(output / "predictions.csv")
    for (fit_id, partition), part in saved.groupby(["fit_id", "partition"]):
        residual = part[["observed_inner", "observed_outer"]].to_numpy() - part[["predicted_inner", "predicted_outer"]].to_numpy()
        expected = metric_table[metric_table.fit_id.eq(fit_id) & metric_table.partition.eq(partition)].iloc[0]
        np.testing.assert_allclose(np.square(residual).mean(), expected.joint_mse, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(residual, part[["corrected_inner", "corrected_outer"]], rtol=1e-12, atol=1e-12)
    primary = saved[saved.max_iter.isna() | saved.max_iter.eq(1500)]
    for (model, seed, partition), part in primary.groupby(["model", "seed", "partition"], dropna=False):
        expected_counts = 1 if partition == "validation" else 3
        assert len(part) == len(rows) * expected_counts
        assert part.groupby("neutral_row").size().eq(expected_counts).all()
    for relative, expected in {**preserved, **source_hashes}.items():
        assert sha256(root / relative) == expected, f"Input/source changed during evaluation: {relative}"
    write_json(output / "verification.json", {
        "passed": True, "recordings": 63, "neutral_frames": 4414, "retained_quality_flags": 65,
        "recording_disjoint_known_signer_folds": 4, "each_recording_validation_once": True,
        "training_only_full_identifier_preprocessing_checked": 8,
        "identical_primary_model_rows_targets": True, "saved_fold_joint_mse_recalculated": True,
        "input_source_hashes_unchanged": True, "frozen_pilot_hashes_checked": len(frozen),
        "upstream_assets_checked": len(upstream), "convergence_remedies_used": len(remedies),
        "prediction_rows": len(saved), "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12})
    write_json(output / "run_metadata.json", {
        "completed_at_oslo": now(), "phase": 5 if save_bundles else 4, "config": config, "folds": folds,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")},
        "threadpools_during_fit": pools, "inputs": data.provenance, "source_sha256": source_hashes,
        "artifact_sha256": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
        "weighting": "training equal frames; validation primary equal recordings pooled across folds; frame-weighted score separate",
        
        "same_process_bundle_verified": bool(save_bundles), "inference_bundle_verified": False,
        "interpretation": "Neutral-measurement prediction on held-out recordings of known signers in the released dataset only."})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/openface/recording-split")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--save-bundles", action="store_true", help="Save and check complete inference bundles")
    args = parser.parse_args()
    summary = run_recording_evaluation(args.output, args.root, save_bundles=args.save_bundles)
    print("PASS: fixed whole-recording folds, training-only refits, identical retained rows and saved scoring.")
    print(summary.to_string(index=False))
    print("Saved:", args.output.resolve())
    print("Evaluation complete; inference bundles saved and checked." if args.save_bundles else "Evaluation complete.")


if __name__ == "__main__":
    main()
