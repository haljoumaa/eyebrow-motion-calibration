"""Independently recalculate saved pilot errors, AUROC and aggregation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from verification_output import fresh_output


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("raw", "gam", "mlp", "mlp_seed_7", "mlp_seed_19")


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def verify(root=ROOT, output=None):
    root = Path(root).resolve()
    out = fresh_output(output, root)
    required = ['results/face-landmarker/predictions.csv',
                'results/face-landmarker/recording_metrics.csv',
                'results/face-landmarker/summary.csv', 'data/manifest.csv']
    missing = [p for p in required if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError('Pilot arithmetic verification requires local-only artifacts: ' +
                                ', '.join(missing) + '. See docs/reproduction.md.')
    predictions = pd.read_csv(root / "results/face-landmarker/predictions.csv")
    metrics = pd.read_csv(root / "results/face-landmarker/recording_metrics.csv")
    summary = pd.read_csv(root / "results/face-landmarker/summary.csv")
    manifest = pd.read_csv(root / "data/manifest.csv").set_index("recording_id")
    checked = []
    annotated = neutral = raised = 0
    for rid, part in predictions.groupby("recording_id", sort=False):
        condition = manifest.loc[rid, "eyebrow_condition"]
        cache = pd.read_csv(root / f"data/cache/{rid}_measurements.csv")
        np.testing.assert_array_equal(part.frame_index, cache.frame_index)
        np.testing.assert_array_equal(part.reference_frame, cache.reference_frame)
        np.testing.assert_array_equal(part.valid_geometry, cache.valid_geometry)
        close(part.timestamp_s, cache.timestamp_s)
        close(part.raw_height, cache.inner_brow_height)
        reference = float(cache.loc[cache.reference_frame & cache.valid_geometry, "inner_brow_height"].median())
        valid = part.valid_geometry & part.raw_height.notna() & part.relative_pitch_deg.notna()
        supported = valid & part.supported_pose
        labelled = supported & ~part.reference_frame & part.visible_state.notna()
        if condition == "alternating":
            annotated += int(labelled.sum())
            neutral += int((labelled & part.visible_state.eq(0)).sum())
            raised += int((labelled & part.visible_state.eq(1)).sum())
        for method in METHODS:
            values = part[method + "_height"]
            for scope, availability in (("supported_pose", supported), ("complete_clip_raw_fallback", valid)):
                saved = metrics.loc[metrics.recording_id.eq(rid) & metrics.method.eq(method) & metrics.scope.eq(scope)]
                if len(saved) != 1:
                    raise ValueError("Missing or repeated pilot metric row")
                saved = saved.iloc[0]
                mask = availability & ~part.reference_frame & values.notna()
                close(saved.reference_height, reference)
                close(saved.detection_coverage, cache.detected.mean())
                close(saved.supported_fraction_all_frames, supported.mean())
                if condition == "alternating":
                    a = values[mask & part.visible_state.eq(1)].to_numpy()
                    b = values[mask & part.visible_state.eq(0)].to_numpy()
                    auc = float((a[:, None] > b).mean() + .5 * (a[:, None] == b).mean())
                    close(saved.auc, auc)
                    if (saved.n_neutral, saved.n_raised, saved.n) != (len(b), len(a), len(a) + len(b)):
                        raise ValueError("Pilot annotation counts disagree")
                else:
                    residual = values[mask].to_numpy() - reference
                    close(saved.mae, np.abs(residual).mean())
                    close(saved.rmse, np.sqrt(np.square(residual).mean()))
                    if saved.n != len(residual):
                        raise ValueError("Pilot stability count disagrees")
                checked.append(saved.to_dict())
    if (len(predictions), predictions.recording_id.nunique(), annotated, neutral, raised) != (3433, 22, 419, 254, 165):
        raise ValueError("Pilot recording/frame totals disagree")
    expected = pd.DataFrame(checked).groupby(["distance", "condition", "method", "scope"], dropna=False)[
        ["rmse", "mae", "auc", "detection_coverage", "supported_fraction_all_frames"]].mean().reset_index()
    pd.testing.assert_frame_equal(summary, expected, rtol=1e-12, atol=1e-12)
    primary = metrics.loc[metrics.scope.eq("supported_pose")]
    values = {}
    for condition, field in (("neutral", "rmse"), ("raised", "rmse"), ("alternating", "auc")):
        values[condition] = {method: float(primary.loc[primary.condition.eq(condition) & primary.method.eq(method), field].mean()) for method in METHODS}
    record = {
        "passed": True, "decoded_frames": len(predictions), "recordings": predictions.recording_id.nunique(),
        "metric_rows_checked": len(checked), "summary_rows_checked": len(summary),
        "supported_annotated_frames": annotated, "neutral_frames": neutral, "raised_frames": raised,
        "pairwise_auroc_recalculated": True, "recording_errors_recalculated": True,
        "all_methods_and_both_scopes_checked": True, "equal_recording_primary_values": values,
        "rtol": 1e-12, "atol": 1e-12,
    }
    if out is not None:
        with (out / "pilot-arithmetic.json").open("x") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
    print(f"PASS: {len(predictions)} pilot frames, {len(checked)} recording metrics, "
          f"{len(summary)} summary rows")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="Optional fresh directory; default: stdout only")
    args = parser.parse_args()
    verify(args.root, args.output)
