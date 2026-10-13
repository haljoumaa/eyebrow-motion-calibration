import numpy as np

from facial_motion.audit import aligned_overlap


def test_reencoded_trimmed_sequence_is_flagged_at_correct_offset():
    rng = np.random.default_rng(42)
    original = rng.uniform(20, 230, (80, 8, 8)).astype(np.float32)
    reencoded = original[17:62] + rng.normal(0, .3, (45, 8, 8))
    result = aligned_overlap(original, reencoded)
    assert result["frame_a"] - result["frame_b"] == 17
    assert result["longest_low_error_run_frames"] == 45


def test_independent_sequences_do_not_trigger_long_match():
    rng = np.random.default_rng(7)
    a, b = rng.uniform(20, 230, (2, 80, 8, 8)).astype(np.float32)
    result = aligned_overlap(a, b)
    assert result["longest_low_error_run_frames"] == 0
    assert result["best_window_mae_gray"] > 40
