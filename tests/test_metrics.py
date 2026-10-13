import numpy as np
import pytest

from facial_motion.metrics import preservation_auc, stability, state_labels


def test_flat_signal_fails_preservation_sanity_check():
    labels = [0, 0, 1, 1]
    assert preservation_auc([2, 2, 2, 2], labels)["auc"] == .5
    assert preservation_auc([0, 0, 1, 1], labels)["auc"] == 1


def test_missing_frames_and_identical_evaluation_masks():
    raw = np.array([1., 2., np.nan, 10.])
    corrected = np.array([1., 1., np.nan, 10.])
    mask = np.array([True, True, True, False])
    assert stability(raw, 1, mask) == {"n": 2, "mae": .5, "rmse": np.sqrt(.5)}
    assert stability(corrected, 1, mask)["rmse"] == 0


def test_manual_intervals_exclude_uncertain_and_transitions():
    intervals = [{"start_s": 0, "end_s": 1, "state": "neutral", "uncertain": False},
                 {"start_s": 2, "end_s": 3, "state": "raised", "uncertain": False},
                 {"start_s": 3, "end_s": 4, "state": "raised", "uncertain": True}]
    labels = state_labels([.5, 1.5, 2.5, 3.5], intervals)
    np.testing.assert_allclose(labels, [0, np.nan, 1, np.nan], equal_nan=True)
    assert np.isnan(preservation_auc([1, 2], [0, 0])["auc"])


def test_overlapping_annotations_rejected():
    intervals = [{"start_s": 0, "end_s": 2, "state": "neutral", "uncertain": False},
                 {"start_s": 1, "end_s": 3, "state": "raised", "uncertain": False}]
    with pytest.raises(ValueError, match="overlap"):
        state_labels([0, 1, 2], intervals)
