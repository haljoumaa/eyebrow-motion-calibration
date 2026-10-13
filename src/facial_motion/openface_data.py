"""Pinned released OpenFace inputs; separate from the Face Landmarker pilot."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = _SOURCE_ROOT if (_SOURCE_ROOT / "config.yaml").exists() else Path.cwd()
TARGETS = ("inner_perp_dist39_42_3d", "outer_perp_dist39_42_3d")
POSE_COLUMNS = ("pose_Rx", "pose_Ry", "pose_Rz", "pose_Tx", "pose_Ty", "pose_Tz")
TABLE_PATH = "research/upstream-review/github/processing/saved/open_face_with_distances.csv"
ANNOTATIONS_PATH = "research/upstream-review/github/processing/saved/additional_stats_from_elan.json"
COMMIT = "c44f4ba6d3b385cfb210fc6df80c8d2c00ef27f0"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class NeutralData:
    rows: pd.DataFrame
    provenance: dict


def load_neutral_data(root=ROOT):
    """Retain source order/quality flags; fail rather than silently repair or filter."""
    root = Path(root)
    manifest_path = root / "research/upstream-assets.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["github_commit"] != COMMIT:
        raise ValueError("Upstream commit differs from the approved protocol")
    assets = {asset["local_path"]: asset for asset in manifest["assets"]}
    hashes = {"research/upstream-assets.json": sha256(manifest_path)}
    for relative in (TABLE_PATH, ANNOTATIONS_PATH):
        if relative not in assets:
            raise ValueError(f"Missing pinned input in asset manifest: {relative}")
        actual = sha256(root / relative)
        if actual != assets[relative]["sha256"]:
            raise ValueError(f"Pinned input hash mismatch: {relative}")
        hashes[relative] = actual

    released = pd.read_csv(root / TABLE_PATH, index_col=[0, 1])
    required = ["speaker", "sentence", "sType", "timestamp", "confidence", "success", *POSE_COLUMNS, *TARGETS]
    missing = set(required) - set(released.columns)
    if missing:
        raise ValueError(f"Missing released columns: {sorted(missing)}")
    if released.index.names != ["video_name", "frame"] or not released.index.is_unique:
        raise ValueError("Expected unique (video_name, frame) source identities")
    annotations = json.loads((root / ANNOTATIONS_PATH).read_text())
    mask = released.index.isin(annotations["videos_no_brows"], level=0) & released.sType.eq("st")
    neutral = released.loc[mask, required].copy()
    neutral.insert(0, "source_row", np.flatnonzero(mask))
    rows = neutral.reset_index()
    rows.insert(0, "neutral_row", np.arange(len(rows)))
    if (len(rows), rows.video_name.nunique(), rows.speaker.nunique()) != (4414, 63, 8):
        raise ValueError("Expected the approved 4,414-frame / 63-recording / eight-signer subset")
    for column in ["video_name", "speaker", "sentence"]:
        if rows[column].isna().any() or not rows[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Missing or invalid identifiers: {column}")
    numeric = rows[["frame", "timestamp", "confidence", "success", *POSE_COLUMNS, *TARGETS]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Released subset contains nonfinite numeric inputs/targets/flags")
    if not rows.confidence.between(0, 1).all() or not rows.success.isin([0, 1]).all():
        raise ValueError("Unexpected confidence/success quality values")
    if not np.equal(rows.frame, np.floor(rows.frame)).all() or rows.frame.lt(0).any():
        raise ValueError("Frame identifiers must be nonnegative integers")
    if rows.groupby("video_name")[["speaker", "sentence"]].nunique().gt(1).any().any():
        raise ValueError("A recording has inconsistent signer/sentence identifiers")
    rows["quality_low_confidence"] = rows.confidence.lt(0.8)
    if int(rows.quality_low_confidence.sum()) != 65:
        raise ValueError("Expected all 65 released low-confidence flags to be retained")
    provenance = {
        "github_commit": COMMIT, "input_sha256": hashes,
        "filter": "video_name in videos_no_brows AND sType == 'st'",
        "released_frames": len(released), "neutral_frames": len(rows),
        "neutral_recordings": 63, "neutral_signers": 8,
        "low_confidence_rows_retained": 65, "success_zero_rows_retained": int(rows.success.eq(0).sum()),
        "target_order": list(TARGETS), "numeric_pose_order": list(POSE_COLUMNS),
        "source_order_preserved": True, "excluded_for_quality": 0,
        "quality_note": "Retained flags do not establish whether coordinates were previously repaired.",
    }
    return NeutralData(rows, provenance)
