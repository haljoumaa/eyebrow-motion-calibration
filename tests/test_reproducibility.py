import numpy as np
import pandas as pd
import pytest

from facial_motion.evaluate import run_evaluation
from facial_motion.reproducibility import compare_table, compare_values


def test_pilot_replay_overwrite_rejected_before_reading_or_fitting(tmp_path):
    output = tmp_path / "kept"
    output.mkdir()
    (output / "sentinel").write_text("preserved")
    with pytest.raises(FileExistsError):
        run_evaluation(root=tmp_path / "no_inputs", output=output)
    assert (output / "sentinel").read_text() == "preserved"


def test_comparison_detects_numerical_and_identity_changes_and_missingness(tmp_path):
    table = pd.DataFrame({"row": [0, 1], "model": ["ridge", "mlp"], "score": [1., np.nan]})
    left, right = tmp_path / "left.csv", tmp_path / "right.csv"
    table.to_csv(left, index=False)
    table.assign(score=[1 + 1e-13, np.nan]).to_csv(right, index=False)
    assert compare_table(left, right)["rows"] == 2
    for bad in (table.iloc[::-1], table.assign(score=[1., 0.]), table.assign(score=[1.01, np.nan])):
        bad.to_csv(right, index=False)
        with pytest.raises(AssertionError):
            compare_table(left, right)
    assert compare_values({"a": [1., "x", True]}, {"a": [1 + 1e-13, "x", True]}) < 1e-12
    with pytest.raises(AssertionError):
        compare_values({"seed": 42}, {"seed": 7})
    with pytest.raises(AssertionError):
        compare_values({"a": True}, {"a": 1})
