"""Native-rate video inference with a fresh tracker and auditable raw outputs."""

import importlib.metadata
import json
import platform
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yaml

from .acquire import ROOT, read_manifest, sha256
from .features import derive_table


def load_config(path=None):
    return yaml.safe_load(Path(path or ROOT / "config.yaml").read_text())


def extract_recording(recording_id, root=ROOT, config=None, reference=None, overlay=False):
    root = Path(root)
    config = config or load_config(root / "config.yaml")
    row = next((r for r in read_manifest(root) if r["recording_id"] == recording_id), None)
    if row is None:
        raise ValueError(f"Unknown recording ID: {recording_id}")
    video = root / row["local_path"]
    if not video.exists():
        raise FileNotFoundError(f"Acquire recording first: {recording_id}")
    actual_hash = sha256(video)
    if row["sha256"] and actual_hash != row["sha256"]:
        raise ValueError("Video checksum differs from manifest")
    model = root / config["model"]["path"]
    asset_metadata = root / "models/model_metadata.json"
    if asset_metadata.exists():
        expected_model_hash = json.loads(asset_metadata.read_text())["sha256"]
        if sha256(model) != expected_model_hash:
            raise ValueError("Model checksum differs from recorded model asset")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid source frame rate")
    advertised_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=config["model"]["num_faces"],
        min_face_detection_confidence=config["model"]["min_face_detection_confidence"],
        min_face_presence_confidence=config["model"]["min_face_presence_confidence"],
        min_tracking_confidence=config["model"]["min_tracking_confidence"],
        output_facial_transformation_matrixes=True, output_face_blendshapes=False,
    )
    landmarks, matrices, timestamps = [], [], []
    previous_ms = -1
    writer = None
    out = root / "results/pilot"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                height, width = image.shape[:2]
                pts = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                timestamp_ms = int(round(pts))
                if not np.isfinite(pts) or timestamp_ms <= previous_ms:
                    raise ValueError("Decoder timestamps are nonmonotonic or collide at millisecond precision; inspect source")
                previous_ms = timestamp_ms
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                    timestamp_ms,
                )
                points = np.full((478, 3), np.nan, dtype=np.float32)
                matrix = np.full((4, 4), np.nan)
                if result.face_landmarks:
                    returned = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32)
                    if returned.shape != points.shape:
                        raise ValueError(f"Unexpected landmark topology: {returned.shape}")
                    points = returned
                    if result.facial_transformation_matrixes:
                        matrix = np.asarray(result.facial_transformation_matrixes[0])
                landmarks.append(points)
                matrices.append(matrix)
                timestamps.append(pts / 1000)
                if overlay:
                    if writer is None:
                        writer = cv2.VideoWriter(str(out / f"{recording_id}_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if not writer.isOpened():
                            raise ValueError("Cannot create overlay video")
                    painted = draw_landmarks(image, points, config, pts / 1000)
                    writer.write(painted)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not timestamps:
        raise ValueError("No frames decoded")
    elapsed = time.perf_counter() - started
    arrays = np.asarray(landmarks), np.asarray(matrices), np.asarray(timestamps)
    cache = root / "data/cache"
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache / f"{recording_id}_landmarks.npz", landmarks=arrays[0], matrices=arrays[1],
                        timestamps=arrays[2], width=width, height=height)
    table, ref_metadata = derive_table(*arrays, recording_id, width, height, config, reference)
    table.to_csv(cache / f"{recording_id}_measurements.csv", index=False)
    metadata = {
        "recording_id": recording_id, "original_filename": row["original_filename"], "video_sha256": actual_hash,
        "model_sha256": sha256(model), "model_url": config["model"]["url"], "config": config,
        "platform": platform.platform(), "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ["mediapipe", "opencv-contrib-python", "numpy", "scipy", "pandas"]},
        "fps": fps, "width": width, "height": height, "advertised_frame_count": advertised_count,
        "decoded_frame_count": len(table), "last_timestamp_s": float(arrays[2][-1]),
        "timestamp_source": "OpenCV CAP_PROP_POS_MSEC after decoding; no inferred timestamp fallback",
        "elapsed_s": elapsed, "inference_frames_per_s": len(table) / elapsed,
        "detection_coverage": float(table.detected.mean()), "valid_geometry_coverage": float(table.valid_geometry.mean()),
        "frame_count_matches": len(table) == advertised_count, **ref_metadata,
    }
    (out / f"{recording_id}_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: metadata[k] for k in ["recording_id", "decoded_frame_count", "detection_coverage", "elapsed_s", "reference_applied"]}, indent=2))
    return table, metadata


def draw_landmarks(image, points, config, timestamp):
    image = image.copy()
    height, width = image.shape[:2]
    palette = {"inner_brows": (0, 220, 255), "outer_brows": (0, 200, 0), "inner_eyes": (255, 190, 0), "scale_eyes": (230, 0, 230)}
    for group, indices in config["landmarks"].items():
        for index in indices:
            if not np.isfinite(points[index]).all():
                continue
            x, y = (points[index, :2] * [width, height]).astype(int)
            cv2.circle(image, (x, y), 5, palette[group], -1)
            cv2.putText(image, str(index), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, palette[group], 2)
    cv2.rectangle(image, (0, 0), (width, 52), (20, 20, 20), -1)
    cv2.putText(image, f"t={timestamp:.2f}s | candidate landmark QA | model estimates", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


def rederive_recording(recording_id, root=ROOT, reference=None):
    """Apply a reviewed reference to cached estimates without another tracker run."""
    root = Path(root)
    config = load_config(root / "config.yaml")
    if reference is None:
        import csv
        with (root / "data/references.csv").open(newline="") as f:
            refs = {r["recording_id"]: r for r in csv.DictReader(f)}
        ref = refs.get(recording_id)
        if ref is None or ref["verified"].lower() != "true":
            raise ValueError("A reviewed reference is required to rederive measurements")
        reference = (float(ref["start_s"]), float(ref["end_s"]))
    metadata_path = root / f"results/pilot/{recording_id}_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    # Changing inference settings requires extraction, rather than relabelling a cache.
    if metadata["config"]["model"] != config["model"]:
        raise ValueError("Inference settings changed; extract the recording again")
    with np.load(root / f"data/cache/{recording_id}_landmarks.npz") as raw:
        table, ref_metadata = derive_table(raw["landmarks"], raw["matrices"], raw["timestamps"],
                                          recording_id, int(raw["width"]), int(raw["height"]), config, reference)
    table.to_csv(root / f"data/cache/{recording_id}_measurements.csv", index=False)
    metadata.update(config=config, **ref_metadata)
    metadata["valid_geometry_coverage"] = float(table.valid_geometry.mean())
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return table, metadata
