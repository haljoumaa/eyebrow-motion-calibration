import pytest

from facial_motion.validate import assert_disjoint, inner_folds, outer_folds


def recordings():
    return [{"recording_id": f"{d}_{s}", "distance": d, "eyebrow_condition": s,
             "duplicate_group": f"{d}_{s}", "sha256": f"hash_{d}_{s}"}
            for d in ["close", "middle", "far"] for s in ["neutral", "raised", "alternating"]]


def test_whole_recordings_and_distance_exclusion():
    for held_out, train, test in outer_folds(recordings()):
        assert all(r["distance"] != held_out and r["eyebrow_condition"] == "neutral" for r in train)
        assert all(r["distance"] == held_out for r in test)
        for fit, validation in inner_folds(train):
            assert {r["distance"] for r in fit}.isdisjoint({r["distance"] for r in validation})


@pytest.mark.parametrize("key", ["recording_id", "duplicate_group", "sha256"])
def test_duplicate_overlap_blocks_partition(key):
    rows = recordings()
    left, right = rows[0].copy(), rows[-1].copy()
    right[key] = left[key]
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint([left], [right])
