"""Geometry in aspect-corrected relative coordinates, never millimetres."""

import warnings

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def aspect_correct(points, width, height):
    return np.asarray(points, dtype=float) * np.array([width, height, width])


def point_line_distances(points, line_start, line_end):
    points = np.asarray(points, dtype=float)
    a, b = np.asarray(line_start), np.asarray(line_end)
    axis = b - a
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Degenerate eye-corner line")
    offsets = points - a
    residuals = offsets - np.outer(offsets @ (axis / norm), axis / norm)
    return np.linalg.norm(residuals, axis=-1)


def measurements(points, indices):
    needed = sum(indices.values(), [])
    if not np.isfinite(np.asarray(points)[needed]).all():
        raise ValueError("Nonfinite landmarks")
    a, b = np.asarray(points)[indices["inner_eyes"]]
    inner = point_line_distances(np.asarray(points)[indices["inner_brows"]], a, b).mean()
    outer = point_line_distances(np.asarray(points)[indices["outer_brows"]], a, b).mean()
    s1, s2 = np.asarray(points)[indices["scale_eyes"]]
    scale = np.linalg.norm(s2 - s1)
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Degenerate scale eye-corner line")
    return float(inner), float(outer), float(scale)


def matrix_rotation(matrix):
    """Polar decomposition removes scale; a reflection/singular matrix is invalid."""
    a = np.asarray(matrix, dtype=float)[:3, :3]
    if not np.isfinite(a).all() or np.linalg.det(a) <= 1e-10:
        raise ValueError("Invalid or reflected transformation matrix")
    u, _, vt = np.linalg.svd(a)
    return u @ vt


def orientation(matrix, reference_rotation=None, pitch_sign=1):
    r = matrix_rotation(matrix)
    if reference_rotation is not None:
        r = np.asarray(reference_rotation).T @ r
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Lowercase xyz means extrinsic rotations: R = Rz(roll) Ry(yaw) Rx(pitch).
        angles = Rotation.from_matrix(r).as_euler("xyz", degrees=True)
    angles[0] *= pitch_sign
    return angles


def derive_table(landmarks, matrices, timestamps, recording_id, width, height, config, reference=None):
    count = len(timestamps)
    distances = np.full((count, 3), np.nan)
    rotations = np.full((count, 3, 3), np.nan)
    angles = np.full((count, 3), np.nan)
    detected = np.isfinite(landmarks).any(axis=(1, 2))
    valid = np.zeros(count, dtype=bool)
    reasons = []
    for i in range(count):
        if not detected[i]:
            reasons.append("no_detection")
            continue
        try:
            distances[i] = measurements(aspect_correct(landmarks[i], width, height), config["landmarks"])
            rotations[i] = matrix_rotation(matrices[i])
            angles[i] = orientation(matrices[i], pitch_sign=config["orientation"]["pitch_sign"])
            valid[i] = True
            reasons.append("")
        except (ValueError, UserWarning) as error:
            reasons.append(str(error))
    reference_mask = np.zeros(count, dtype=bool)
    heights = np.full((count, 2), np.nan)
    relative_angles = np.full((count, 3), np.nan)
    metadata = {"reference_applied": False, "reference_scale": None}
    if reference is not None:
        start, end = reference
        if end - start < config["reference"]["minimum_duration_s"] - 1e-8:
            raise ValueError("Reference is shorter than the documented minimum")
        reference_mask = (timestamps >= start) & (timestamps < end)
        usable_reference = reference_mask & valid
        if usable_reference.sum() < config["reference"]["minimum_valid_frames"]:
            raise ValueError("Reference contains too few valid frames")
        scale = np.median(distances[usable_reference, 2])
        ref_rotation = Rotation.from_matrix(rotations[usable_reference]).mean().as_matrix()
        heights[valid] = distances[valid, :2] / scale
        for i in np.flatnonzero(valid):
            try:
                relative_angles[i] = orientation(matrices[i], ref_rotation, config["orientation"]["pitch_sign"])
            except (ValueError, UserWarning) as error:
                valid[i] = False
                heights[i] = np.nan
                reasons[i] = str(error)
        metadata.update(reference_applied=True, reference_start_s=start, reference_end_s=end,
                        reference_scale=float(scale), reference_valid_frames=int(usable_reference.sum()),
                        reference_rotation=ref_rotation.tolist())
    frame = pd.DataFrame({
        "recording_id": recording_id, "frame_index": np.arange(count), "timestamp_s": timestamps,
        "detected": detected, "pitch_deg": angles[:, 0], "yaw_deg": angles[:, 1], "roll_deg": angles[:, 2],
        "relative_pitch_deg": relative_angles[:, 0], "relative_yaw_deg": relative_angles[:, 1],
        "relative_roll_deg": relative_angles[:, 2], "inner_brow_distance_px": distances[:, 0],
        "outer_brow_distance_px": distances[:, 1], "eye_scale_px": distances[:, 2],
        "inner_brow_height": heights[:, 0], "outer_brow_height": heights[:, 1],
        "reference_frame": reference_mask, "valid_geometry": valid, "quality_reason": reasons,
    })
    return frame, metadata
