"""Scientific guards for the fixed OpenFace model reproduction."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ShuffleSplit

from facial_motion.openface_data import POSE_COLUMNS, TARGETS, load_neutral_data
from facial_motion.openface_reproduction import (
    FIXED_CONFIG, OpenFacePreprocessor, audit_preprocessing, fit_estimator,
    make_estimator, numeric_features, run_reproduction,
)


@pytest.fixture
def training():
    # Encounter order deliberately differs from alphabetical order.
    rows = pd.DataFrame({"speaker": ["z", "a", "z", "a"], "sentence": ["last", "first", "first", "last"],
                         "pose_Rx": [0., .1, .2, .3], "pose_Ry": [.4, .5, .6, .7],
                         "pose_Rz": [.8, .9, 1., 1.1], "pose_Tx": [10., 20., 30., 40.],
                         "pose_Ty": [-10., -20., -30., -40.], "pose_Tz": [100., 200., 300., 400.],
                         TARGETS[0]: [2., 3., 4., 6.], TARGETS[1]: [12., 15., 21., 33.]})
    return rows


def test_radian_cosines_and_archived_interleaved_numeric_order():
    rows = pd.DataFrame([[0, np.pi / 2, np.pi, 10, 20, 30]], columns=POSE_COLUMNS)
    np.testing.assert_allclose(numeric_features(rows), [[0, 1, 10, np.pi / 2, 0, 20, np.pi, -1, 30]], atol=1e-15)


def test_categories_scaler_and_one_hot_columns_are_training_only(training):
    prep = OpenFacePreprocessor().fit(training)
    state = prep.state()
    assert state["categories"] == {"speaker": ["a", "z"], "sentence": ["first", "last"]}
    assert state["feature_order"][-4:] == ["speaker_a", "speaker_z", "sentence_first", "sentence_last"]
    np.testing.assert_array_equal(prep.design(training)[:, -4:], [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]])
    validation = training.iloc[:2].copy()
    validation.loc[:, "pose_Tz"] = 1000000
    validation.loc[:, "sentence"] = "unseen"
    validation.loc[:, list(TARGETS)] = -999999
    prep.transform(validation)
    assert prep.state() == state
    assert prep.unknown_sentence_mask(validation).tolist() == [True, True]
    np.testing.assert_array_equal(prep.design(validation)[:, -2:], np.zeros((2, 2)))
    assert state["scaler_n_samples_seen"] == 4
    assert state["scaler_mean"][8] == 250
    assert state["scaler_var"][8] == 12500


def test_predictions_accept_no_targets_and_ignore_changed_validation_targets(training):
    validation = training.iloc[[1, 3]].copy()
    prep = OpenFacePreprocessor().fit(training)
    x = prep.transform(training)
    model = make_estimator("ridge").fit(x, training[list(TARGETS)].to_numpy())
    original = model.predict(prep.transform(validation))
    validation.loc[:, list(TARGETS)] += [100000, -200000]
    np.testing.assert_array_equal(model.predict(prep.transform(validation)), original)
    np.testing.assert_array_equal(model.predict(prep.transform(validation.drop(columns=list(TARGETS)))), original)
    np.testing.assert_array_equal(model.predict(prep.transform(validation.iloc[::-1])), original[::-1])


def test_joint_ridge_matches_independent_penalized_solution(training):
    prep = OpenFacePreprocessor().fit(training)
    x = prep.transform(training)
    y = training[list(TARGETS)].to_numpy()
    x_centered, y_mean = x - x.mean(axis=0), y.mean(axis=0)
    weights = np.linalg.solve(x_centered.T @ x_centered + .001 * np.eye(x.shape[1]), x_centered.T @ (y - y_mean))
    expected = x_centered @ weights + y_mean
    actual = make_estimator("ridge").fit(x, y).predict(x)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert actual.shape == (4, 2)  # inner first, outer second, in original units
    assert actual[:, 1].mean() > 5 * actual[:, 0].mean()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_is_rejected_without_row_drops(training, bad):
    training.loc[0, "pose_Rx"] = bad
    with pytest.raises(ValueError, match="finite"):
        OpenFacePreprocessor().fit(training)


def test_unknown_signer_missing_columns_and_invalid_identifier_fail(training):
    prep = OpenFacePreprocessor().fit(training)
    unknown = training.iloc[:1].copy()
    unknown.loc[:, "speaker"] = "not-trained"
    with pytest.raises(ValueError, match="Unknown signer"):
        prep.transform(unknown)
    with pytest.raises(ValueError, match="Missing OpenFace input columns"):
        prep.transform(training.drop(columns="pose_Ry"))
    with pytest.raises(ValueError, match="Missing OpenFace input columns"):
        prep.transform(training.drop(columns="sentence"))
    with pytest.raises(ValueError, match="nonempty"):
        prep.fit(training.iloc[:0])
    unknown.loc[:, "sentence"] = None
    with pytest.raises(ValueError, match="nonempty strings"):
        prep.fit(unknown)


def test_estimator_configuration_and_seed_policy_are_fixed():
    assert make_estimator("ridge").alpha == .001
    for seed in (42, 7, 19):
        model = make_estimator("mlp", seed)
        assert (model.hidden_layer_sizes, model.activation, model.solver) == ((40,), "relu", "adam")
        assert (model.alpha, model.max_iter, model.early_stopping, model.random_state) == (.0001, 1500, False, seed)
        assert (model.tol, model.n_iter_no_change, model.learning_rate_init, model.batch_size) == (.0001, 10, .001, "auto")
        assert make_estimator("mlp", seed, 3000).max_iter == 3000
    with pytest.raises(ValueError, match="fixed"):
        make_estimator("mlp", 999)
    with pytest.raises(ValueError, match="fixed"):
        make_estimator("mlp", 42, 9000)


def test_real_historical_folds_scaler_and_legacy_values():
    from facial_motion.openface_data import ROOT, TABLE_PATH, ANNOTATIONS_PATH
    if any(not (ROOT / p).is_file() for p in (TABLE_PATH, ANNOTATIONS_PATH)):
        pytest.skip('External OpenFace inputs absent; run python scripts/fetch_data.py --group openface')
    rows = load_neutral_data().rows
    assert rows.quality_low_confidence.sum() == 65
    for train_ids, validation_ids in ShuffleSplit(**FIXED_CONFIG["splitter"]).split(rows):
        training, validation = rows.iloc[train_ids], rows.iloc[validation_ids]
        assert (len(training), len(validation)) == (3310, 1104)
        assert set(training.video_name) & set(validation.video_name) == set(rows.video_name)
        prep = OpenFacePreprocessor().fit(training)
        audit = audit_preprocessing(prep, training, validation, rows)
        assert audit["legacy_numerical_matrix_identical"]
        assert audit["design_width"] == 27
        assert audit["unknown_validation_sentences"] == 0


def test_convergence_warning_is_retained_and_invalidates_fit():
    class IncompleteEstimator:
        coef_ = np.zeros((2, 1))
        intercept_ = np.zeros(2)
        n_iter_ = 1500

        def fit(self, x, y):
            warnings.warn("test incomplete optimization", ConvergenceWarning)
            return self

        def get_params(self, deep=False):
            return {"max_iter": 1500}

    state = fit_estimator(IncompleteEstimator(), np.zeros((2, 1)), np.zeros((2, 2)), "incomplete-example")
    assert not state["fit_complete"]
    assert state["n_iter"] == 1500
    assert state["warnings"][0]["category"] == "ConvergenceWarning"
    assert state["warnings"][0]["message"] == "test incomplete optimization"


def test_runner_rejects_existing_output_before_any_input_or_fit(tmp_path):
    (tmp_path / "sentinel").write_text("preserve")
    with pytest.raises(FileExistsError, match="already exists"):
        run_reproduction(tmp_path, root=tmp_path / "missing-root")
    assert (tmp_path / "sentinel").read_text() == "preserve"
