"""Compare isolated experiment replays and check every bundle in a fresh process."""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from .openface_bundle import load_bundle
from .openface_data import ROOT, TARGETS, sha256
from .openface_reproduction import now, write_json


RTOL = ATOL = 1e-12
VOLATILE_FIT_KEYS = {"fit_seconds", "completed_at_oslo"}


def compare_values(left, right, location="root"):
    """Check exact structure/identities/settings and tolerant floating-point state."""
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), f"Keys differ: {location}"
        return max((compare_values(left[k], right[k], f"{location}.{k}") for k in left), default=0.)
    if isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right), f"List differs: {location}"
        return max((compare_values(a, b, f"{location}[{i}]") for i, (a, b) in enumerate(zip(left, right))), default=0.)
    if isinstance(left, float):
        assert isinstance(right, (float, int)) and not isinstance(right, bool), location
        np.testing.assert_allclose(left, right, rtol=RTOL, atol=ATOL, err_msg=location)
        return float(abs(left - right))
    assert type(left) is type(right) and left == right, f"Value differs: {location}: {left!r} / {right!r}"
    return 0.


def compare_table(left_path, right_path):
    left, right = pd.read_csv(left_path), pd.read_csv(right_path)
    assert left.columns.tolist() == right.columns.tolist(), f"Columns differ: {left_path.name}"
    assert left.shape == right.shape, f"Shape differs: {left_path.name}"
    maximum = 0.
    for column in left:
        a, b = left[column], right[column]
        assert a.isna().equals(b.isna()), f"Missingness differs: {left_path.name}.{column}"
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
            np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True,
                                       err_msg=f"{left_path.name}.{column}")
            diff = np.abs(a.to_numpy(dtype=float) - b.to_numpy(dtype=float))
            if np.isfinite(diff).any():
                maximum = max(maximum, float(np.nanmax(diff)))
        else:
            pd.testing.assert_series_equal(a, b, check_exact=True)
    return {"file": left_path.name, "rows": len(left), "columns": len(left.columns),
            "max_absolute_numeric_difference": maximum}


def _read(path):
    return json.loads(Path(path).read_text())


def _without_fit_times(runs):
    return [{k: v for k, v in run.items() if k not in VOLATILE_FIT_KEYS} for run in runs]


def verify_fresh_process(root, output, replay=None):
    replay = Path(replay) if replay is not None else root / "results/openface/recording-split"
    out = output / "fresh-process"
    out.mkdir(exist_ok=False)
    rows = pd.read_csv(replay / "neutral_rows.csv")
    source = pd.read_csv(replay / "predictions.csv")
    prep = _read(replay / "preprocessing.json")
    means = _read(replay / "training_means.json")
    runs = {r["fit_id"]: r for r in _read(replay / "model_runs.json")}
    manifest = _read(replay / "bundle_manifest.json")
    assert len(manifest["bundles"]) == 28 and manifest["same_process_predictions_checked"]
    permutation = np.random.default_rng(19).permutation(len(rows))
    pd.DataFrame({"input_position": np.arange(len(rows)), "neutral_row": rows.iloc[permutation].neutral_row.to_numpy()}).to_csv(out / "input_row_order.csv", index=False)
    checks, commands = [], []
    for item in manifest["bundles"]:
        fit_id = item["fit_id"]
        path = replay / item["path"]
        bundle = load_bundle(path, item["sha256"])
        fold = str(item["fold"])
        expected_prep = {k: v for k, v in prep[fold]["identifiers" if item["model"] == "identifier_ridge" else "full"].items() if k != "audit"}
        assert bundle.payload["preprocessing"] == expected_prep
        state = bundle.payload["estimator"]
        if item["model"] in {"global", "signer"}:
            assert state["parameters"] == means[fold][item["model"]]
        else:
            run = runs[fit_id]
            compare_values(json.loads(json.dumps(run["resolved_estimator_parameters"])), state["parameters"], fit_id)
            if item["model"] == "mlp":
                assert state["coefs"] == run["coefs"] and state["intercepts"] == run["intercepts"]
                assert state["n_iter"] == run["n_iter"]
                losses = pd.read_csv(replay / "loss_curves.csv")
                np.testing.assert_allclose(state["loss_curve"], losses.loc[losses.fit_id.eq(fit_id), "training_loss"], rtol=RTOL, atol=ATOL)
            else:
                assert state["coef"] == run["coefs"] and state["intercept"] == run["intercepts"]
        assert bundle.payload["context"]["config"] == _read(root / "configs/openface-recording-evaluation.json")
        assignment = pd.read_csv(replay / "assignments.csv")
        assigned = assignment[assignment.fold.eq(item["fold"])]
        for partition in ("training", "validation"):
            assert bundle.payload["context"][f"{partition}_rows"] == assigned.loc[assigned.partition.eq(partition), "neutral_row"].tolist()
        # Move each bundle and pass only input columns, in shuffled row/column order.
        relocated = out / "bundles" / path.name
        relocated.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, relocated)
        inputs = out / f"{fit_id}-inputs.csv"
        rows.iloc[permutation][bundle.required[::-1]].to_csv(inputs, index=False)
        target = out / f"{fit_id}-predictions.csv"
        command = [sys.executable, "-I", "-B", "-m", "facial_motion.openface_bundle",
                   "--bundle", str(relocated), "--input", str(inputs), "--output", str(target),
                   "--expected-sha256", item["sha256"]]
        result = subprocess.run(command, cwd=out, capture_output=True, text=True)
        commands.append({"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        write_json(out / "process-log.json", commands)
        assert result.returncode == 0, result.stderr
        actual = pd.read_csv(target)
        expected = source[source.fit_id.eq(fit_id)].set_index("neutral_row").loc[rows.iloc[permutation].neutral_row]
        assert len(expected) == len(rows)
        for column in ("predicted_inner", "predicted_outer"):
            np.testing.assert_allclose(actual[column], expected[column], rtol=RTOL, atol=ATOL)
        for column in ("unknown_sentence", "unknown_signer"):
            np.testing.assert_array_equal(actual[column], expected[column])
        np.testing.assert_array_equal(actual.input_position, np.arange(len(rows)))
        same = bundle.predict(rows.iloc[permutation][bundle.required[::-1]])
        np.testing.assert_allclose(same.iloc[:, :2], actual[["predicted_inner", "predicted_outer"]], rtol=RTOL, atol=ATOL)
        errors = np.abs(actual[["predicted_inner", "predicted_outer"]].to_numpy() - expected[["predicted_inner", "predicted_outer"]].to_numpy())
        checks.append({"fit_id": fit_id, "frames": len(actual), "training_frames": int(expected.partition.eq("training").sum()),
            "validation_frames": int(expected.partition.eq("validation").sum()), "fresh_process_exit_code": result.returncode,
            "max_absolute_prediction_difference": float(errors.max()), "bundle_state_matches_fitted_artifacts": True,
            "input_only_and_row_column_reordering_checked": True, "relocated_bundle_sha256": sha256(relocated),
            "input_sha256": sha256(inputs), "output_sha256": sha256(target)})
        print(f"PASS fresh-process {fit_id}: {len(actual)} permuted input-only rows", flush=True)
    write_json(output / "bundle-verification.json", {"passed": True, "checked_at_oslo": now(),
        "bundles": checks, "fresh_processes": len(checks), "prediction_rows": sum(c["frames"] for c in checks),
        "target_order": list(TARGETS), "rtol": RTOL, "atol": ATOL,
        "max_absolute_prediction_difference": max(c["max_absolute_prediction_difference"] for c in checks),
        "no_upstream_pickles_or_retraining_on_load": True})


