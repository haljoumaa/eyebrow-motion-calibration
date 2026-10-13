import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from facial_motion.openface_baselines import MeanBaseline
from facial_motion.openface_bundle import load_bundle, predict_csv, save_bundle
from facial_motion.openface_data import POSE_COLUMNS, TARGETS, sha256
from facial_motion.openface_recording import IdentifierPreprocessor
from facial_motion.openface_reproduction import OpenFacePreprocessor, make_estimator


@pytest.fixture
def fitted_bundle(tmp_path):
    def fit(model):
        n = 60
        rows = pd.DataFrame({"speaker": ["A", "B"] * (n // 2),
            "sentence": ["first", "second", "third"] * (n // 3),
            **{c: np.sin(np.arange(n) / (i + 2)) for i, c in enumerate(POSE_COLUMNS)}})
        rows[TARGETS[0]] = 2 + rows.pose_Rx * .2
        rows[TARGETS[1]] = 4 + rows.pose_Ry * .3
        prep = (IdentifierPreprocessor() if model == "identifier_ridge" else OpenFacePreprocessor()).fit(rows)
        with threadpool_limits(limits=1):
            estimator = (MeanBaseline(model).fit(rows) if model in {"global", "signer"} else
                make_estimator("ridge" if model == "identifier_ridge" else model, 42 if model == "mlp" else None)
                .fit(prep.transform(rows), rows[list(TARGETS)]))
        context = {"fit_id": "test", "fold": 0, "config": {"test": True},
                   "input_provenance": {"test": "synthetic"}, "source_sha256": {}}
        path = tmp_path / f"{model}.json"
        bundle = save_bundle(path, prep, estimator, model, context)
        return path, bundle, prep, estimator, rows
    return fit


@pytest.mark.parametrize("model", ["global", "signer", "ridge", "identifier_ridge", "mlp"])
def test_bundle_matches_fitted_estimator_after_fresh_process_and_permutation(fitted_bundle, tmp_path, model):
    path, bundle, prep, estimator, rows = fitted_bundle(model)
    inputs = rows[bundle.required].copy()
    with threadpool_limits(limits=1):
        expected = estimator.predict(rows.speaker) if model in {"global", "signer"} else estimator.predict(prep.transform(rows))
    loaded = load_bundle(path, sha256(path))
    np.testing.assert_allclose(loaded.predict(inputs).iloc[:, :2], expected, rtol=1e-12, atol=1e-12)
    order = np.random.default_rng(19).permutation(len(inputs))
    shuffled = inputs.iloc[order, ::-1]
    same_process = loaded.predict(shuffled)
    np.testing.assert_allclose(same_process.iloc[:, :2], expected[order], rtol=1e-12, atol=1e-12)
    assert same_process.index.tolist() == shuffled.index.tolist()
    input_path, output_path = tmp_path / "inputs.csv", tmp_path / "predictions.csv"
    shuffled.to_csv(input_path, index=False)
    run = subprocess.run([sys.executable, "-B", "-m", "facial_motion.openface_bundle",
        "--bundle", str(path), "--input", str(input_path), "--output", str(output_path),
        "--expected-sha256", sha256(path)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    actual = pd.read_csv(output_path)
    np.testing.assert_allclose(actual[["predicted_inner", "predicted_outer"]], expected[order], rtol=1e-12, atol=1e-12)
    assert not actual.unknown_sentence.any() and not actual.unknown_signer.any()
    assert actual.input_position.tolist() == list(range(len(inputs)))
    changed = rows.copy()
    changed[list(TARGETS)] = np.nan
    pd.testing.assert_frame_equal(bundle.predict(changed), loaded.predict(inputs))


@pytest.mark.parametrize("model", ["global", "signer", "ridge", "identifier_ridge", "mlp"])
def test_unseen_sentences_flagged_unknown_signers_rejected(fitted_bundle, model):
    path, bundle, prep, estimator, rows = fitted_bundle(model)
    unseen = rows.iloc[:3].assign(sentence="unseen")
    result = load_bundle(path).predict(unseen)
    assert result.unknown_sentence.all()
    with threadpool_limits(limits=1):
        expected = estimator.predict(unseen.speaker) if model in {"global", "signer"} else estimator.predict(prep.transform(unseen))
    np.testing.assert_allclose(result.iloc[:, :2], expected, rtol=1e-12, atol=1e-12)
    if model not in {"global", "signer"}:
        count = len(bundle.categories["sentence"])
        np.testing.assert_array_equal(bundle.design(unseen)[:, -count:], np.zeros((3, count)))
        np.testing.assert_allclose(bundle.transform(unseen)[:, -count:],
            np.broadcast_to(-bundle.mean[-count:] / bundle.scale[-count:], (3, count)))
    with pytest.raises(ValueError, match="Unknown signer"):
        bundle.predict(rows.iloc[:1].assign(speaker="unknown"))


@pytest.mark.parametrize("problem, message", [
    ("missing", "Missing OpenFace input"), ("duplicate", "Duplicate"),
    ("empty", "nonempty"), ("nonfinite", "finite"), ("nonnumeric", "numeric"),
    ("missing_category", "nonempty strings"), ("category_number", "nonempty strings")])
def test_invalid_prediction_inputs_have_useful_errors(fitted_bundle, problem, message):
    _, bundle, _, _, rows = fitted_bundle("ridge")
    bad = rows.copy()
    if problem == "missing":
        bad = bad.drop(columns="pose_Rx")
    elif problem == "duplicate":
        bad = pd.concat([bad, bad[["speaker"]]], axis=1)
    elif problem == "empty":
        bad = bad.iloc[:0]
    elif problem in {"nonfinite", "nonnumeric"}:
        bad["pose_Rx"] = np.inf if problem == "nonfinite" else "bad"
    else:
        bad["sentence"] = None if problem == "missing_category" else 123
    with pytest.raises(ValueError, match=message):
        bundle.predict(bad)


@pytest.mark.parametrize("problem", ["version", "targets", "features", "categories", "scale", "weights", "runtime", "missing"])
def test_malformed_or_incompatible_bundle_rejected(fitted_bundle, problem):
    path, bundle, _, _, _ = fitted_bundle("ridge")
    bad = copy.deepcopy(bundle.payload)
    if problem == "version":
        bad["schema_version"] = 999
    elif problem == "targets":
        bad["target_order"].reverse()
    elif problem == "features":
        bad["preprocessing"]["feature_order"].reverse()
    elif problem == "categories":
        bad["preprocessing"]["categories"]["speaker"].append("A")
    elif problem == "scale":
        bad["preprocessing"]["scaler_scale"][0] = 0
    elif problem == "weights":
        bad["estimator"]["coef"][0].pop()
    elif problem == "runtime":
        bad["runtime"]["packages"]["numpy"] = "incompatible"
    else:
        del bad["context"]
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_bundle(path)


def test_hash_and_overwrite_guards_protect_existing_outputs(fitted_bundle, tmp_path):
    path, _, prep, estimator, rows = fitted_bundle("ridge")
    with pytest.raises(ValueError, match="SHA-256"):
        load_bundle(path, "0" * 64)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_bundle(path, prep, estimator, "ridge", {})
    assert path.read_bytes() == before
    output = tmp_path / "kept.csv"
    output.write_text("sentinel")
    with pytest.raises(FileExistsError):
        predict_csv(path, tmp_path / "nonexistent.csv", output)
    assert output.read_text() == "sentinel"
    inputs = tmp_path / "duplicate.csv"
    inputs.write_text("speaker,speaker,sentence\nA,A,first\n")
    with pytest.raises(ValueError, match="Duplicate"):
        predict_csv(path, inputs, tmp_path / "new.csv")
