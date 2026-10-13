"""Independent recording audit: direct saved-weight predictions and table arithmetic."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from facial_motion.openface_data import TARGETS, load_neutral_data, sha256
from protocol_hashes import verify_source_hashes
from verification_output import fresh_output

ROOT = Path(__file__).resolve().parents[1]
SCORES = ["joint_mse", "inner_mse", "inner_rmse", "inner_mae", "outer_mse", "outer_rmse", "outer_mae"]


def arithmetic(group):
    residual = group[["observed_inner", "observed_outer"]].to_numpy() - group[["predicted_inner", "predicted_outer"]].to_numpy()
    result = {"n_frames": len(group), "joint_mse": float(np.square(residual).mean())}
    for i, label in enumerate(("inner", "outer")):
        mse = float(np.mean(residual[:, i] ** 2))
        result.update({f"{label}_mse": mse, f"{label}_rmse": float(np.sqrt(mse)),
                       f"{label}_mae": float(np.mean(np.abs(residual[:, i])))})
    return result


def select(table, model, seed, limit):
    result = table[table.model.eq(model)]
    result = result[result.seed.isna() if pd.isna(seed) else result.seed.eq(seed)]
    return result[result.max_iter.isna() if pd.isna(limit) else result.max_iter.eq(limit)]


def check(root=ROOT, output=None):
    root = Path(root).resolve()
    out = fresh_output(output, root)
    evaluation = root / "results/openface/recording-split"
    revisions = {}
    for experiment in ("baselines", "frame-split", "recording-split"):
        run = json.loads((root / f"results/openface/{experiment}/run_metadata.json").read_text())
        revisions[experiment] = verify_source_hashes(root, run["source_sha256"])
    rows = load_neutral_data(root).rows.set_index("neutral_row", drop=False)
    allocation = pd.read_csv(root / "configs/recording-folds.csv")
    metadata = json.loads((evaluation / "run_metadata.json").read_text())
    config = json.loads((root / "configs/openface-recording-evaluation.json").read_text())
    assert metadata["config"] == config
    assert sha256(root / config["fold_path"]) == config["fold_sha256"]
    assert (config["primary_seed"], config["sensitivity_seeds"]) == (42, [7, 19])
    assert not config["target_scaling"] and config["quality_exclusions"] == 0
    assert config["ridge"]["alpha"] == config["identifier_ridge"]["alpha"] == .001
    omitted = {}
    for p, h in metadata["artifact_sha256"].items():
        path = evaluation / p
        if path.is_file():
            assert sha256(path) == h, p
        else:
            # The original run records the full generated output. Only these
            # redundant fit copies and prior verification summaries may be absent.
            allowed = p in {"recording_allocation.csv", "verification.json"} or p.startswith("fit_states/") or (
                p.startswith("bundles/") and p != "bundles/fold0_ridge.json")
            assert allowed, p
            omitted[p] = h
    if (evaluation / "recording_allocation.csv").exists():
        pd.testing.assert_frame_equal(pd.read_csv(evaluation / "recording_allocation.csv"), allocation, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_csv(evaluation / "neutral_rows.csv"), rows.reset_index(drop=True),
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    assignments = pd.read_csv(evaluation / "assignments.csv")
    predictions = pd.read_csv(evaluation / "predictions.csv")
    recs = pd.read_csv(evaluation / "recording_metrics.csv")
    fold_metrics = pd.read_csv(evaluation / "fold_metrics.csv")
    summary = pd.read_csv(evaluation / "summary.csv")
    signers = pd.read_csv(evaluation / "signer_metrics.csv")
    pairs = pd.read_csv(evaluation / "paired_recording_differences.csv")
    curves = pd.read_csv(evaluation / "loss_curves.csv")
    prep = json.loads((evaluation / "preprocessing.json").read_text())
    means = json.loads((evaluation / "training_means.json").read_text())
    coverage = json.loads((evaluation / "unknown_sentence_coverage.json").read_text())
    runs = json.loads((evaluation / "model_runs.json").read_text())
    weights_by_id = {run["fit_id"]: run for run in runs}
    remedies = json.loads((evaluation / "convergence_remedies.json").read_text())
    assert not remedies  # Observed delivery has no incomplete fits/remedies.
    assert len(runs) == 20 and sum(r["model"] == "mlp" for r in runs) == 12
    assert len(weights_by_id) == len(runs)
    assert all(r["fit_complete"] and not r["warnings"] for r in runs)
    assert (len(predictions), len(assignments), len(recs), len(fold_metrics), len(summary), len(signers), len(pairs)) == (123592, 17656, 1764, 56, 7, 56, 819)
    assert predictions.fit_complete.all() and not predictions.unknown_signer.any()
    max_prediction_error, max_metric_error = 0., 0.
    fold_records, independent_recs = [], []
    with threadpool_limits(limits=1):
        for fold in range(4):
            held = set(allocation.loc[allocation.validation_fold.eq(fold), "video_name"])
            validation_ids = rows.index[rows.video_name.isin(held)].to_numpy()
            training_ids = rows.index[~rows.video_name.isin(held)].to_numpy()
            training = rows.loc[training_ids]
            validation = rows.loc[validation_ids]
            assert set(training.video_name).isdisjoint(validation.video_name)
            assert set(validation.speaker) <= set(training.speaker)
            assert int(training.quality_low_confidence.sum() + validation.quality_low_confidence.sum()) == 65

            def design(part, state, numeric):
                blocks = []
                if numeric:
                    for axis in "xyz":
                        angle = part[f"pose_R{axis}"].to_numpy(dtype=float)
                        blocks.extend([angle, np.cos(angle), part[f"pose_T{axis}"].to_numpy(dtype=float)])
                for category in ("speaker", "sentence"):
                    blocks.extend([part[category].eq(value).to_numpy(dtype=float) for value in state["categories"][category]])
                return np.column_stack(blocks)

            for key, numeric in (("full", True), ("identifiers", False)):
                state = prep[str(fold)][key]
                assert state["scaler_n_samples_seen"] == len(training)
                assert state["target_order"] == list(TARGETS)
                for cat in ("speaker", "sentence"):
                    assert state["categories"][cat] == sorted(training[cat].unique())
                matrix = design(training, state, numeric)
                expected_names = [item for axis in "xyz" for item in (f"pose_R{axis}", f"pose_R{axis}_cos", f"pose_T{axis}")] if numeric else []
                expected_names += [f"{cat}_{value}" for cat in ("speaker", "sentence") for value in state["categories"][cat]]
                assert state["feature_order"] == expected_names
                assert len(state["feature_order"]) == (27 if numeric else 18)
                np.testing.assert_allclose(state["scaler_mean"], matrix.mean(axis=0), rtol=1e-12, atol=1e-12)
                variance = matrix.var(axis=0)
                np.testing.assert_allclose(state["scaler_var"], variance, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(state["scaler_scale"], np.where(variance == 0, 1., np.sqrt(variance)), rtol=1e-12, atol=1e-12)
                assert not np.allclose(state["scaler_mean"], design(rows, state, numeric).mean(axis=0), rtol=0, atol=1e-12)
            for name in ("global", "signer"):
                saved_means = means[str(fold)][name]
                assert saved_means["n_training_frames"] == len(training)
                np.testing.assert_allclose(saved_means["global_mean"], training[list(TARGETS)].mean().to_numpy(), rtol=1e-12, atol=1e-12)
                for speaker, part in training.groupby("speaker"):
                    np.testing.assert_allclose(saved_means["signer_means"][speaker], part[list(TARGETS)].mean().to_numpy(), rtol=1e-12, atol=1e-12)

            for partition, ids in (("training", training_ids), ("validation", validation_ids)):
                part = rows.loc[ids]
                assigned = assignments[assignments.fold.eq(fold) & assignments.partition.eq(partition)]
                np.testing.assert_array_equal(assigned.neutral_row, ids)
                unknown = ~part.sentence.isin(training.sentence.unique()).to_numpy()
                np.testing.assert_array_equal(assigned.unknown_sentence, unknown)
                cov = [c for c in coverage if c["fold"] == fold and c["partition"] == partition][0]
                assert cov["n_frames"] == len(part) and cov["n_recordings"] == part.video_name.nunique()
                assert cov["unknown_sentence_frames"] == int(unknown.sum())
                assert cov["unknown_sentence_recordings"] == part.loc[unknown].video_name.nunique()
                assert cov["unknown_sentences"] == sorted(part.loc[unknown].sentence.unique())
                current = predictions[predictions.fold.eq(fold) & predictions.partition.eq(partition)]
                assert current.fit_id.nunique() == 7
                for fit_id, group in current.groupby("fit_id", sort=False):
                    assert not group.neutral_row.duplicated().any()
                    np.testing.assert_array_equal(group.neutral_row, ids)
                    for key in ("source_row", "video_name", "frame", "speaker", "sentence", "success", "quality_low_confidence"):
                        np.testing.assert_array_equal(group[key], part[key])
                    np.testing.assert_allclose(group.confidence, part.confidence, rtol=0, atol=0)
                    np.testing.assert_allclose(group[["observed_inner", "observed_outer"]], part[list(TARGETS)], rtol=1e-12, atol=1e-12)
                    np.testing.assert_array_equal(group.unknown_sentence, unknown)
                    predicted = group[["predicted_inner", "predicted_outer"]].to_numpy()
                    model = group.model.iloc[0]
                    if model in ("ridge", "identifier_ridge", "mlp"):
                        weights = weights_by_id[fit_id]
                        state = prep[str(fold)]["identifiers" if model == "identifier_ridge" else "full"]
                        z = (design(part, state, model != "identifier_ridge") - np.asarray(state["scaler_mean"])) / np.asarray(state["scaler_scale"])
                        params = weights["resolved_estimator_parameters"]
                        if model == "mlp":
                            assert params["hidden_layer_sizes"] == [40]
                            assert (params["activation"], params["solver"], params["alpha"], params["max_iter"], params["early_stopping"]) == ("relu", "adam", .0001, 1500, False)
                            assert params["random_state"] == int(group.seed.iloc[0])
                            assert sum(np.asarray(v).size for v in weights["coefs"] + weights["intercepts"]) == 1202
                            hidden = np.maximum(0, z @ np.asarray(weights["coefs"][0]) + np.asarray(weights["intercepts"][0]))
                            independent = hidden @ np.asarray(weights["coefs"][1]) + np.asarray(weights["intercepts"][1])
                            curve = curves[curves.fit_id.eq(fit_id)]
                            assert len(curve) == weights["n_iter"] < 1500
                            np.testing.assert_array_equal(curve.iteration, np.arange(1, len(curve) + 1))
                            assert np.isfinite(curve.training_loss).all()
                        else:
                            assert params["alpha"] == .001
                            independent = z @ np.asarray(weights["coefs"]).T + np.asarray(weights["intercepts"])
                    elif model == "global":
                        independent = np.tile(training[list(TARGETS)].mean().to_numpy(), (len(part), 1))
                    else:
                        independent = np.array([training[training.speaker.eq(s)][list(TARGETS)].mean().to_numpy() for s in part.speaker])
                    max_prediction_error = max(max_prediction_error, float(np.abs(predicted - independent).max()))
                    np.testing.assert_allclose(predicted, independent, rtol=1e-12, atol=1e-12)
                    residual = part[list(TARGETS)].to_numpy() - predicted
                    np.testing.assert_allclose(group[["corrected_inner", "corrected_outer"]], residual, rtol=1e-12, atol=1e-12)
                    calculated = arithmetic(group)
                    saved_metric = fold_metrics[fold_metrics.fit_id.eq(fit_id) & fold_metrics.partition.eq(partition)].iloc[0]
                    for name, value in calculated.items():
                        np.testing.assert_allclose(saved_metric[name], value, rtol=1e-12, atol=1e-12)
                        max_metric_error = max(max_metric_error, abs(float(saved_metric[name]) - value))
                    for video, video_group in group.groupby("video_name", sort=False):
                        calc = arithmetic(video_group)
                        saved_rec = recs[recs.fit_id.eq(fit_id) & recs.partition.eq(partition) & recs.video_name.eq(video)].iloc[0]
                        for name, value in calc.items():
                            np.testing.assert_allclose(saved_rec[name], value, rtol=1e-12, atol=1e-12)
                            max_metric_error = max(max_metric_error, abs(float(saved_rec[name]) - value))
                        assert saved_rec.unknown_sentence_frames == int(video_group.unknown_sentence.sum())
                        assert saved_rec.low_confidence_frames == int(video_group.quality_low_confidence.sum())
                        independent_recs.append({**saved_rec[["model", "seed", "max_iter", "fold", "partition", "video_name", "speaker", "sentence"]].to_dict(), **calc})
            fold_records.append({"fold": fold, "training_frames": len(training), "validation_frames": len(validation),
                "training_recordings": training.video_name.nunique(), "validation_recordings": validation.video_name.nunique(),
                "shared_recordings": 0, "unknown_signers": 0,
                "unknown_sentence_frames": int((~validation.sentence.isin(training.sentence)).sum())})

    independent_recs = pd.DataFrame(independent_recs)
    held = independent_recs[independent_recs.partition.eq("validation")]
    for record in summary.to_dict("records"):
        group = select(held, record["model"], record["seed"], record["max_iter"])
        assert len(group) == record["n_recordings"] == group.video_name.nunique() == 63
        assert group.n_frames.sum() == record["n_frames"] == 4414
        assert record["n_folds"] == 4 and record["all_fits_complete"] and record["eligible_four_fold_comparison"]
        for col in SCORES:
            np.testing.assert_allclose(record[f"equal_recording_{col}"], group[col].sum() / 63, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(record["frame_weighted_joint_mse"], (group.joint_mse * group.n_frames).sum() / 4414, rtol=1e-12, atol=1e-12)
        assert record["unknown_sentence_frames"] == record["unknown_sentence_recordings"] == 0
    for record in signers.to_dict("records"):
        group = select(held, record["model"], record["seed"], record["max_iter"])
        group = group[group.speaker.eq(record["speaker"])]
        assert len(group) == record["n_recordings"] and group.n_frames.sum() == record["n_frames"]
        for col in SCORES:
            np.testing.assert_allclose(record[f"equal_recording_{col}"], group[col].mean(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(record["frame_weighted_joint_mse"], (group.joint_mse * group.n_frames).sum() / group.n_frames.sum(), rtol=1e-12, atol=1e-12)
    for record in pairs.to_dict("records"):
        candidate = select(held, record["candidate"], record["seed"], record["max_iter"])
        control = select(held, record["control"], np.nan, np.nan)
        a = candidate[candidate.video_name.eq(record["video_name"])].iloc[0]
        b = control[control.video_name.eq(record["video_name"])].iloc[0]
        assert a.n_frames == b.n_frames == record["n_frames"]
        assert a.fold == b.fold == record["fold"] and a.speaker == b.speaker == record["speaker"]
        np.testing.assert_allclose([record["candidate_joint_mse"], record["control_joint_mse"], record["joint_mse_difference"]], [a.joint_mse, b.joint_mse, a.joint_mse - b.joint_mse], rtol=1e-12, atol=1e-12)
        assert record["candidate_improves"] == (a.joint_mse < b.joint_mse)
    for (model, seed), group in predictions[predictions.partition.eq("validation")].groupby(["model", "seed"], dropna=False):
        np.testing.assert_array_equal(np.sort(group.neutral_row), np.arange(4414))
    primary_pairs = pairs[pairs.candidate.eq("mlp") & pairs.seed.eq(42) & pairs.control.eq("ridge")]
    write = {"passed": True, "neutral_frames": 4414, "recordings": 63, "signers": 8, "retained_quality_flags": 65,
        "prediction_rows_checked": len(predictions), "recording_metric_rows_checked": len(recs),
        "fold_metric_rows_checked": len(fold_metrics), "summary_rows_checked": len(summary),
        "signer_metric_rows_checked": len(signers), "paired_difference_rows_checked": len(pairs),
        "training_only_preprocessors_checked": 8, "folds": fold_records,
        "independent_predictions_from_saved_weights": True,
        "max_absolute_saved_weight_prediction_difference": max_prediction_error,
        "max_absolute_metric_difference": max_metric_error,
        "equal_recording_and_frame_aggregates_recalculated": True,
        "all_models_identical_rows_targets_flags_order": True,
        "each_recording_validation_once": True, "mlp_fits": 12, "ridge_fits": 8,
        "fit_warnings": 0, "convergence_remedies_used": 0,
        "unknown_sentence_validation_frames": 0, "unknown_sentence_validation_recordings": 0,
        "retained_artifact_hashes_match": True,
        "source_hashes_checked_with_recorded_revisions": revisions,
        "omitted_regenerable_artifact_sha256": omitted,
        "tolerances": {"relative": 1e-12, "absolute": 1e-12}}
    # Retain descriptive gains/failures for the review, without changing any settings.
    comparisons = []
    for keys, group in pairs.groupby(["candidate", "seed", "control"], dropna=False, sort=False):
        candidate, seed, control = keys
        comparisons.append({"candidate": candidate, "seed": None if pd.isna(seed) else int(seed), "control": control,
            "n_recordings": len(group), "improved_recordings": int(group.candidate_improves.sum()),
            "worsened_recordings": int(group.joint_mse_difference.gt(0).sum()),
            "tied_recordings": int(group.joint_mse_difference.eq(0).sum()),
            "mean_joint_mse_difference": float(group.joint_mse_difference.mean()),
            "median_joint_mse_difference": float(group.joint_mse_difference.median()),
            "minimum_joint_mse_difference": float(group.joint_mse_difference.min()),
            "maximum_joint_mse_difference": float(group.joint_mse_difference.max())})
    inspection = {"descriptive_only": True, "comparisons": comparisons,
        "primary_mlp_vs_ridge_largest_failures": primary_pairs.nlargest(3, "joint_mse_difference").to_dict("records"),
        "primary_mlp_vs_ridge_best_differences": primary_pairs.nsmallest(3, "joint_mse_difference").to_dict("records")}
    if out is not None:
        with (out / "verification.json").open("x") as stream:
            stream.write(json.dumps(write, indent=2) + "\n")
        with (out / "result_inspection.json").open("x") as stream:
            stream.write(json.dumps(inspection, indent=2, allow_nan=False) + "\n")
    print(f"PASS: {len(predictions)} OpenFace predictions, {len(recs)} recording metrics; "
          "recorded sources and saved-result arithmetic")
    return write


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="Optional fresh directory; default: stdout only")
    args = parser.parse_args()
    check(args.root, args.output)
