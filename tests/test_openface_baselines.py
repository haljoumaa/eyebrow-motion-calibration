"""Check OpenFace mean baselines, source rows, split assignments and score arithmetic."""

import json

import numpy as np
import pandas as pd
import pytest

from facial_motion.openface_baselines import MeanBaseline, run_baselines, score_targets
from facial_motion.openface_data import ROOT, TABLE_PATH, ANNOTATIONS_PATH, TARGETS, load_neutral_data


@pytest.fixture
def training():
    return pd.DataFrame({"speaker": ["A", "A", "A", "A", "B"],
                         "video_name": ["short", "long", "long", "long", "other"],
                         TARGETS[0]: [2., 6., 6., 6., 10.], TARGETS[1]: [4., 8., 8., 8., 14.]})


def test_hand_means_are_frame_weighted_not_recording_weighted(training):
    global_model = MeanBaseline("global").fit(training)
    signer_model = MeanBaseline("signer").fit(training)
    np.testing.assert_allclose(global_model.predict(["A", "B"]), [[6, 8.4], [6, 8.4]], atol=1e-12)
    np.testing.assert_array_equal(signer_model.predict(["A", "B"]), [[5, 7], [10, 14]])
    assert signer_model.parameters()["signer_counts"] == {"A": 4, "B": 1}
    # Averaging A's two recording means would give [4, 6], which is wrong here.
    assert not np.array_equal(signer_model.predict(["A"])[0], [4, 6])


@pytest.mark.parametrize("mode,expected", [("global", 4.58), ("signer", 4.0)])
def test_hand_joint_mse_and_subtracted_measurement(training, mode, expected):
    predicted = MeanBaseline(mode).fit(training).predict(["A", "B"])
    observed = np.array([[7, 9], [8, 12]])
    assert score_targets(observed, predicted)["joint_mse"] == pytest.approx(expected, abs=1e-12)
    if mode == "signer":
        np.testing.assert_array_equal(observed - predicted, [[2, 2], [-2, -2]])


def test_unknown_signer_has_flagged_global_fallback_and_order_is_preserved(training):
    model = MeanBaseline().fit(training)
    np.testing.assert_allclose(model.predict(["C", "B", "A"]), [[6, 8.4], [10, 14], [5, 7]])
    assert model.unknown_signer_mask(["C", "B", "A"]).tolist() == [True, False, False]
    np.testing.assert_array_equal(model.predict(["A", "B"])[::-1], model.predict(["B", "A"]))


def test_validation_target_mutation_changes_score_but_not_fit(training):
    rows = pd.concat([training, pd.DataFrame({"speaker": ["A", "B"], TARGETS[0]: [7, 8], TARGETS[1]: [9, 12]})], ignore_index=True)
    baseline = MeanBaseline().fit(rows.iloc[:5])
    before = baseline.predict(rows.iloc[5:].speaker)
    changed = rows.copy()
    changed.loc[5:, list(TARGETS)] += 1000
    refitted = MeanBaseline().fit(changed.iloc[:5])
    assert refitted.parameters() == baseline.parameters()
    np.testing.assert_array_equal(refitted.predict(rows.iloc[5:].speaker), before)
    assert score_targets(changed.iloc[5:][list(TARGETS)], before)["joint_mse"] != 4


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_fit_rejects_nonfinite_targets_without_dropping_rows(training, invalid):
    training.loc[0, TARGETS[0]] = invalid
    with pytest.raises(ValueError, match="finite"):
        MeanBaseline().fit(training)


def test_fit_rejects_empty_missing_or_unidentified_training(training):
    with pytest.raises(ValueError, match="empty"):
        MeanBaseline().fit(training.iloc[:0])
    with pytest.raises(ValueError, match="Missing training columns"):
        MeanBaseline().fit(training.drop(columns=TARGETS[1]))
    training.loc[0, "speaker"] = None
    with pytest.raises(ValueError, match="Signer identifiers"):
        MeanBaseline().fit(training)
    with pytest.raises(ValueError, match="Fit the baseline"):
        MeanBaseline().predict(["A"])


def test_corrupt_pinned_input_is_rejected_before_loading(tmp_path):
    (tmp_path / "research").mkdir()
    (tmp_path / "research/upstream-assets.json").write_bytes((ROOT / "research/upstream-assets.json").read_bytes())
    bad = tmp_path / TABLE_PATH
    bad.parent.mkdir(parents=True)
    bad.write_text("corrupt input\n")
    with pytest.raises(ValueError, match="Pinned input hash mismatch"):
        load_neutral_data(tmp_path)


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    if any(not (ROOT / p).is_file() for p in (TABLE_PATH, ANNOTATIONS_PATH)):
        pytest.skip('External OpenFace inputs absent; run python scripts/fetch_data.py --group openface')
    output = tmp_path_factory.mktemp("phase-1-check") / "packet"
    run_baselines(output)
    return output


def test_released_subset_and_quality_fields_retained_exactly(packet):
    rows = pd.read_csv(packet / "neutral_rows.csv")
    released = pd.read_csv(ROOT / TABLE_PATH, index_col=[0, 1])
    selected = released.iloc[rows.source_row.to_numpy()].reset_index()
    assert (len(rows), rows.video_name.nunique(), rows.speaker.nunique()) == (4414, 63, 8)
    assert rows.quality_low_confidence.sum() == 65
    assert rows.success.eq(0).sum() == 65
    columns = ["video_name", "frame", "speaker", "sentence", "confidence", "success", *TARGETS]
    pd.testing.assert_frame_equal(rows[columns], selected[columns], check_exact=False, rtol=1e-12, atol=1e-12)


def test_saved_training_assignments_reproduce_frame_weighted_parameters(packet):
    rows = pd.read_csv(packet / "neutral_rows.csv").set_index("neutral_row")
    assignments = pd.read_csv(packet / "historical_assignments.csv")
    parameters = json.loads((packet / "training_means.json").read_text())
    for fold, part in assignments.groupby("fold"):
        train_ids = part.loc[part.partition.eq("training"), "neutral_row"]
        validation_ids = part.loc[part.partition.eq("validation"), "neutral_row"]
        assert (len(train_ids), len(validation_ids)) == (3310, 1104)
        assert set(train_ids).isdisjoint(validation_ids)
        assert set(train_ids) | set(validation_ids) == set(rows.index)
        training = rows.loc[train_ids]
        assert training.quality_low_confidence.sum() + rows.loc[validation_ids].quality_low_confidence.sum() == 65
        expected_global = training[list(TARGETS)].sum().to_numpy() / len(training)
        for mode in ("global", "signer"):
            state = parameters[str(fold)][mode]
            np.testing.assert_allclose(state["global_mean"], expected_global, rtol=1e-12, atol=1e-12)
            for signer, group in training.groupby("speaker"):
                assert state["signer_counts"][signer] == len(group)
                expected = group[list(TARGETS)].sum().to_numpy() / len(group)
                np.testing.assert_allclose(state["signer_means"][signer], expected, rtol=1e-12, atol=1e-12)


def test_saved_prediction_metrics_target_order_and_holdout_independence(packet):
    predictions = pd.read_csv(packet / "historical_predictions.csv")
    metrics = pd.read_csv(packet / "historical_metrics.csv")
    verification = json.loads((packet / "verification.json").read_text())
    assert len(predictions) == 4 * 2 * 1104
    assert not predictions.unknown_signer.any()
    assert not predictions.used_global_fallback.any()
    for (fold, mode), part in predictions.groupby(["fold", "model"]):
        residual = part[["observed_inner", "observed_outer"]].to_numpy() - part[["predicted_inner", "predicted_outer"]].to_numpy()
        expected = metrics[metrics.fold.eq(fold) & metrics.model.eq(mode) & metrics.partition.eq("validation")].iloc[0]
        np.testing.assert_allclose(residual, part[["corrected_inner", "corrected_outer"]], rtol=1e-12, atol=1e-12)
        assert np.square(residual).sum() / residual.size == pytest.approx(expected.joint_mse, rel=1e-12, abs=1e-12)
        for column, label in enumerate(("inner", "outer")):
            assert np.square(residual[:, column]).mean() == pytest.approx(expected[f"{label}_mse"], rel=1e-12)
            assert np.abs(residual[:, column]).mean() == pytest.approx(expected[f"{label}_mae"], rel=1e-12)
    assert len(verification["held_out_target_mutation_checks"]) == 8
    assert all(check["max_prediction_change"] == check["max_parameter_change"] == 0 for check in verification["held_out_target_mutation_checks"])


def test_runner_rejects_output_overwrite(packet):
    with pytest.raises(FileExistsError, match="already exists"):
        run_baselines(packet)
