import numpy as np
import pandas as pd
import pytest

from facial_motion.evaluate import compare_recording, fit_model, nested_comparison, run_evaluation
from facial_motion.extract import load_config


def synthetic_experiment():
    rows, tables, annotations = [], {}, []
    time = np.arange(80) / 10
    pitch = np.concatenate([np.zeros(5), np.linspace(-15, 15, 75)])
    for d in ["close", "middle", "far"]:
        for state in ["neutral", "raised", "alternating"]:
            rid = f"{d}_{state}"
            row = {"recording_id": rid, "distance": d, "direction": "up", "eyebrow_condition": state,
                   "duplicate_group": rid, "sha256": rid}
            rows.append(row)
            expression = .05 if state == "raised" else np.where(time >= 4, .05, 0.) if state == "alternating" else 0.
            noise = np.where(time < .5, 0., .0005 * np.sin(time * 2.1))
            height = .25 + .002 * pitch + expression + noise
            tables[rid] = pd.DataFrame({"recording_id": rid, "frame_index": np.arange(80), "timestamp_s": time,
                "relative_pitch_deg": pitch, "inner_brow_height": height,
                "reference_frame": np.arange(80) < 5, "valid_geometry": True, "detected": True})
            if state == "alternating":
                annotations.extend([{"recording_id": rid, "start_s": .6, "end_s": 3.8, "state": "neutral", "uncertain": False},
                                    {"recording_id": rid, "start_s": 4.2, "end_s": 7.8, "state": "raised", "uncertain": False}])
    return rows, tables, annotations


def test_nested_driver_preserves_partition_and_reference_levels():
    rows, tables, annotations = synthetic_experiment()
    config = load_config()
    # A single penalty/seed makes this a fast integration test, not a model search.
    config["training"].update(gam_penalties=[.1], mlp_penalties=[.01], mlp_seeds=[42])
    metrics, predictions, decisions = nested_comparison(rows, tables, annotations, config)
    assert len(decisions) == 3
    for fold in decisions:
        assert all(rid.startswith(fold["held_out_distance"] + "_") for rid in fold["test_ids"])
        assert all(not rid.startswith(fold["held_out_distance"] + "_") and rid.endswith("neutral") for rid in fold["neutral_training_ids"])
    assert set(metrics.method) == {"raw", "gam", "mlp"}
    assert metrics.groupby(["recording_id", "scope"]).n.nunique().max() == 1
    assert predictions.gam_height.notna().all()
    raised_reference = predictions[(predictions.recording_id == "far_raised") & predictions.reference_frame]
    np.testing.assert_allclose(raised_reference.gam_height, .30, atol=1e-10)


def test_raised_recording_cannot_train_correction():
    rows, tables, _ = synthetic_experiment()
    with pytest.raises(ValueError, match="neutral"):
        fit_model("gam", .1, [rows[1]], tables, load_config())


def test_real_evaluation_waits_for_protocol_review(tmp_path):
    import yaml
    config = load_config()
    config["protocol_status"] = "draft"
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="draft"):
        run_evaluation(tmp_path)
