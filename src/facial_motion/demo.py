"""Local preview on user-supplied footage. No upload or publication operation."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .acquire import ROOT, MANIFEST_FIELDS, read_manifest, sha256, write_manifest
from .evaluate import fit_model
from .extract import draw_landmarks, extract_recording, load_config


def video_preview(video, output, reference, method="raw", calibration_fold="close", root=ROOT):
    root = Path(root)
    video, output = Path(video).expanduser().resolve(), Path(output).expanduser().resolve()
    if output == video:
        raise ValueError("Preview output must differ from the input video")
    if output.exists():
        raise FileExistsError("Preview exists; choose another output path")
    if not video.is_file():
        raise FileNotFoundError(video)
    config = load_config(root / "config.yaml")
    work = output.parent / (output.stem + "_cache")
    if work.exists():
        raise FileExistsError("Preview cache exists; choose another output path")
    work.mkdir(parents=True, exist_ok=False)
    (work / "data").mkdir()
    (work / "models").mkdir(exist_ok=True)
    (work / "models/model_metadata.json").write_text((root / "models/model_metadata.json").read_text())
    row = dict.fromkeys(MANIFEST_FIELDS, "")
    row.update(recording_id="local_preview", local_path=str(video), original_filename=video.name, sha256=sha256(video), include="false")
    write_manifest([row], work)
    preview_config = {**config, "model": {**config["model"], "path": str(root / config["model"]["path"])}}
    table, metadata = extract_recording("local_preview", work, preview_config, reference)
    corrected, support = table.inner_brow_height.to_numpy(), table.valid_geometry.to_numpy()
    if method != "raw":
        run = json.loads((root / "results/face-landmarker/run_metadata.json").read_text())
        fold = next(f for f in run["folds"] if f["held_out_distance"] == calibration_fold)
        rows = [r for r in read_manifest(root) if r["recording_id"] in fold["neutral_training_ids"]]
        tables = {r["recording_id"]: pd.read_csv(root / f"data/cache/{r['recording_id']}_measurements.csv") for r in rows}
        for rid in tables:
            path = f"data/cache/{rid}_measurements.csv"
            if sha256(root / path) != run["input_hashes"][path]:
                raise ValueError("Calibration inputs changed since the saved experiment")
        spec = fold["selected_models"][method]
        model = fit_model(method, spec["penalty"], rows, tables, run["config"], spec["seed"])
        corrected, support = model.apply(table.relative_pitch_deg, table.inner_brow_height)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("Cannot reopen source video for preview")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), metadata["fps"], (metadata["width"], metadata["height"]))
    if not writer.isOpened():
        cap.release()
        raise ValueError("Cannot create preview output")
    count = 0
    try:
        with np.load(work / "data/cache/local_preview_landmarks.npz") as raw:
            for i, frame_row in table.iterrows():
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Preview decoding no longer matches extraction")
                if frame.shape[:2] != (metadata["height"], metadata["width"]):
                    raise ValueError("Preview dimensions differ from extraction")
                timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not np.isclose(timestamp_s, frame_row.timestamp_s, rtol=0, atol=1e-6):
                    raise ValueError("Preview timestamps differ from extraction")
                painted = draw_landmarks(frame, raw["landmarks"][i], config, frame_row.timestamp_s)
                h, w = painted.shape[:2]
                panel = min(240, max(100, h // 4))
                cv2.rectangle(painted, (0, h-panel), (w, h), (25, 28, 31), -1)
                font = max(.35, min(.75, w/1800))
                raw_text = f"RAW height: {_number(frame_row.inner_brow_height)} | pitch: {_number(frame_row.relative_pitch_deg)} deg (down +)"
                cv2.putText(painted, raw_text, (12, h-panel+25), cv2.FONT_HERSHEY_SIMPLEX, font, (230,230,230), 1)
                if method != "raw":
                    text = f"{method.upper()} preview: {_number(corrected[i])} | " + ("supported pose" if support[i] else "raw fallback / correction unavailable")
                    cv2.putText(painted, text, (12, h-panel+48), cv2.FONT_HERSHEY_SIMPLEX, font, (130,210,230), 1)
                cv2.putText(painted, "Height units: fixed reference eye span | local measurement preview", (12, h-12), cv2.FONT_HERSHEY_SIMPLEX, font*.8, (180,180,180), 1)
                _draw_trace(painted, table.inner_brow_height.to_numpy(), corrected, i, method, (12, h-panel+65, w-12, h-32))
                writer.write(painted)
                count += 1
            if cap.read()[0]:
                raise ValueError("Preview source has more frames than extraction")
    finally:
        cap.release()
        writer.release()
    if count != len(table):
        raise ValueError("Preview frame count differs from extraction")
    (work / "preview_metadata.json").write_text(json.dumps({"input_video_sha256": metadata["video_sha256"],
        "output_video_sha256": sha256(output), "frames": count, "audio_included": False,
        "method": method, "calibration_fold": calibration_fold if method != "raw" else None,
        "reference": list(reference), "interpretation": "Local preview only. Correction can degrade held raised expressions and visible-raise separation; no accuracy or behavioral labels."}, indent=2))
    print(f"Local preview ({count} frames; no audio): {output}")
    return output


def _number(value):
    return f"{value:.4f}" if np.isfinite(value) else "unavailable"


def _draw_trace(image, raw, corrected, current, method, bounds):
    x0,y0,x1,y1=bounds
    all_values=np.r_[raw,corrected]
    finite=all_values[np.isfinite(all_values)]
    if not len(finite) or y1<=y0:
        return
    lo,hi=float(finite.min()),float(finite.max())
    span=max(hi-lo,.001)
    for values,color in [(raw,(190,190,190)),(corrected,(170,170,70))] if method != "raw" else [(raw,(190,190,190))]:
        for j in range(1,current+1):
            if not np.isfinite(values[j-1:j+1]).all():
                continue
            points=[(int(x0+k/max(1,len(raw)-1)*(x1-x0)),int(y1-(values[k]-lo)/span*(y1-y0))) for k in [j-1,j]]
            cv2.line(image,points[0],points[1],color,2)
