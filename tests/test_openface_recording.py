import json

import numpy as np
import pandas as pd
import pytest

from facial_motion.openface_data import POSE_COLUMNS, TARGETS, load_neutral_data
from facial_motion.openface_recording import (FIXED_CONFIG, IdentifierPreprocessor,
    audit_preprocessor, paired_recordings, recording_splits, run_recording_evaluation,
    summarize_recordings)
from facial_motion.openface_reproduction import OpenFacePreprocessor, make_estimator


def small_rows():
    records = []
    for signer in ("A", "B"):
        for fold in range(4):
            for frame in range(fold + 1):
                records.append({"video_name": f"{signer}{fold}", "speaker": signer,
                    "sentence": f"sentence{fold}", "frame": frame,
                    **{c: (fold + frame / 10 + i) / 5 for i, c in enumerate(POSE_COLUMNS)},
                    TARGETS[0]: fold + (0 if signer == "A" else 2) + frame,
                    TARGETS[1]: fold * 2 + frame})
    rows = pd.DataFrame(records)
    allocation = rows[["video_name", "speaker"]].drop_duplicates().copy()
    allocation["validation_fold"] = allocation.video_name.str[-1].astype(int)
    return rows, allocation


def test_recording_folds_cover_rows_once_preserve_order_and_known_signers():
    rows, allocation = small_rows()
    splits = recording_splits(rows, allocation)
    counts = np.zeros(len(rows), dtype=int)
    for fold, (train, validation) in enumerate(splits):
        counts[validation] += 1
        assert (np.diff(train) > 0).all() and (np.diff(validation) > 0).all()
        assert set(rows.iloc[train].video_name).isdisjoint(rows.iloc[validation].video_name)
        assert set(rows.iloc[validation].speaker) <= set(rows.iloc[train].speaker)
        assert set(rows.iloc[validation].video_name) == {f"A{fold}", f"B{fold}"}
    np.testing.assert_array_equal(counts, np.ones(len(rows)))


@pytest.mark.parametrize("problem", ["duplicate", "missing", "extra", "signer", "fold", "unknown_signer"])
def test_invalid_allocations_are_rejected(problem):
    rows, allocation = small_rows()
    if problem == "duplicate":
        allocation = pd.concat([allocation, allocation.iloc[:1]])
    elif problem == "missing":
        allocation = allocation.iloc[1:]
    elif problem == "extra":
        allocation = pd.concat([allocation, pd.DataFrame([{"video_name": "extra", "speaker": "A", "validation_fold": 0}])])
    elif problem == "signer":
        allocation.loc[allocation.index[0], "speaker"] = "wrong"
    elif problem == "fold":
        allocation.loc[allocation.index[0], "validation_fold"] = 4
    else:
        rows.loc[rows.video_name.eq("A0"), "speaker"] = "new"
        allocation.loc[allocation.video_name.eq("A0"), "speaker"] = "new"
    with pytest.raises(ValueError):
        recording_splits(rows, allocation)


def test_identifier_design_cannot_use_pose_or_targets_and_handles_unknown_sentence():
    rows, allocation = small_rows()
    train, val = recording_splits(rows, allocation)[0]
    training, validation = rows.iloc[train], rows.iloc[val]
    prep = IdentifierPreprocessor().fit(training)
    inputs_only = validation[["speaker", "sentence"]]
    changed = validation.copy()
    changed[list(POSE_COLUMNS) + list(TARGETS)] = np.nan
    np.testing.assert_array_equal(prep.transform(validation), prep.transform(inputs_only))
    np.testing.assert_array_equal(prep.transform(validation), prep.transform(changed))
    assert prep.unknown_sentence_mask(validation).all()
    assert prep.state()["numeric_feature_order"] == []
    assert prep.state()["scaler_n_samples_seen"] == len(training)
    assert prep.state()["categories"]["sentence"] == ["sentence1", "sentence2", "sentence3"]
    assert prep.design(validation)[:, 2:].sum() == 0
    np.testing.assert_allclose(prep.transform(validation)[:, 2:],
        np.broadcast_to(-prep.scaler_.mean_[2:] / prep.scaler_.scale_[2:], (len(validation), 3)))
    with pytest.raises(ValueError, match="Unknown signer"):
        prep.transform(inputs_only.assign(speaker="new"))


def test_training_only_scalers_and_held_out_targets_do_not_change_predictions():
    rows, allocation = small_rows()
    train, val = recording_splits(rows, allocation)[3]
    training, validation = rows.iloc[train], rows.iloc[val]
    for cls, numeric in ((OpenFacePreprocessor, True), (IdentifierPreprocessor, False)):
        prep = cls().fit(training)
        assert audit_preprocessor(prep, training, validation, numeric)["passed"]
        estimator = make_estimator("ridge").fit(prep.transform(training), training[list(TARGETS)])
        before = estimator.predict(prep.transform(validation))
        mutated = validation.copy()
        mutated[list(TARGETS)] = [1e12, -1e12]
        np.testing.assert_array_equal(before, estimator.predict(prep.transform(mutated)))
        np.testing.assert_allclose(before[::-1], estimator.predict(prep.transform(mutated.iloc[::-1])), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(estimator.coef_, make_estimator("ridge").fit(cls().fit(training).transform(training), training[list(TARGETS)]).coef_)


def recording_scores():
    # Unequal recording lengths and fold sizes distinguish three possible aggregates.
    return pd.DataFrame({"model": ["ridge"] * 3, "seed": [None] * 3, "max_iter": [None] * 3,
        "partition": ["validation"] * 3, "fold": [0, 0, 1], "video_name": ["a", "b", "c"],
        "speaker": ["A", "A", "B"], "sentence": ["s"] * 3, "n_frames": [1, 9, 2],
        "joint_mse": [1., 9., 4.], "inner_mse": [1., 9., 4.], "outer_mse": [1., 9., 4.],
        "inner_rmse": [1., 3., 2.], "outer_rmse": [1., 3., 2.],
        "inner_mae": [1., 3., 2.], "outer_mae": [1., 3., 2.],
        "fit_complete": [True] * 3, "unknown_sentence_frames": [0, 9, 0]})


def test_primary_aggregation_weights_recordings_not_frames_or_folds():
    scored = recording_scores()
    result = summarize_recordings(scored).iloc[0]
    assert result.equal_recording_joint_mse == pytest.approx(14 / 3)
    assert result.frame_weighted_joint_mse == pytest.approx(90 / 12)
    assert result.equal_recording_inner_rmse == 2  # mean recording RMSE, not sqrt(mean MSE)
    assert result.equal_recording_joint_mse != scored.groupby("fold").joint_mse.mean().mean()
    assert result.unknown_sentence_recordings == 1 and result.unknown_sentence_frames == 9
    assert not result.eligible_four_fold_comparison
    with pytest.raises(ValueError, match="multiple validation"):
        summarize_recordings(pd.concat([scored, scored.iloc[:1]]))


def test_paired_recording_differences_keep_seed_and_sign_of_failures():
    base = recording_scores()
    tables = [base.assign(model=m) for m in ("global", "signer", "ridge", "identifier_ridge")]
    tables += [base.assign(model="mlp", seed=s, max_iter=1500, joint_mse=[.5, 10., 3.]) for s in (42, 7, 19)]
    pairs = paired_recordings(pd.concat(tables))
    assert len(pairs) == 3 * 13
    assert set(pairs[pairs.candidate.eq("mlp")].seed) == {42, 7, 19}
    p = pairs[pairs.candidate.eq("mlp") & pairs.seed.eq(42) & pairs.control.eq("ridge")]
    np.testing.assert_array_equal(p.joint_mse_difference, [-.5, 1., -1.])
    np.testing.assert_array_equal(p.candidate_improves, [True, False, True])
    malformed = pd.concat(tables).loc[lambda x: ~(x.model.eq("signer") & x.video_name.eq("b"))]
    with pytest.raises(ValueError, match="identical recordings"):
        paired_recordings(malformed)


def test_existing_output_is_rejected_before_loading_or_fitting(tmp_path):
    output = tmp_path / "kept"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("preserved")
    with pytest.raises(FileExistsError):
        run_recording_evaluation(output, root=tmp_path / "no_inputs")
    assert sentinel.read_text() == "preserved"


def test_fixed_config_and_real_fold_inventory():
    from facial_motion.openface_data import ROOT, TABLE_PATH, ANNOTATIONS_PATH, sha256
    from facial_motion.openface_recording import CONFIG_PATH, FOLD_PATH, FOLD_SHA256
    assert json.loads((ROOT / CONFIG_PATH).read_text()) == FIXED_CONFIG
    assert sha256(ROOT / FOLD_PATH) == FOLD_SHA256
    if any(not (ROOT / p).is_file() for p in (TABLE_PATH, ANNOTATIONS_PATH)):
        pytest.skip('External OpenFace inputs absent; run python scripts/fetch_data.py --group openface')
    rows = load_neutral_data().rows
    folds = recording_splits(rows, pd.read_csv(ROOT / FOLD_PATH))
    assert [(len(t), len(v)) for t, v in folds] == [(3164, 1250), (3204, 1210), (3434, 980), (3440, 974)]
    assert [rows.iloc[v].video_name.nunique() for _, v in folds] == [17, 17, 15, 14]
