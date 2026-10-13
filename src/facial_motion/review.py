"""Local video QA tools, with raw frames available before seeing model output."""

from pathlib import Path

import cv2
import numpy as np

from .acquire import ROOT, read_manifest


def contact_sheet(recording_id, root=ROOT, times=None, landmarks_path=None, config=None):
    root = Path(root)
    row = next(r for r in read_manifest(root) if r["recording_id"] == recording_id)
    cap = cv2.VideoCapture(str(root / row["local_path"]))
    duration = float(row["duration_s"])
    times = times if times is not None else np.linspace(0, max(0, duration - 0.12), 12)
    cached = np.load(landmarks_path) if landmarks_path else None
    tiles = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if cached is not None:
                from .extract import draw_landmarks
                idx = np.argmin(np.abs(cached["timestamps"] - actual))
                frame = draw_landmarks(frame, cached["landmarks"][idx], config, actual)
            h, w = frame.shape[:2]
            # A central crop makes face details readable; source frames remain untouched.
            crop = frame[int(h * .10):int(h * .85), int(w * .20):int(w * .80)]
            tile = cv2.resize(crop, (480, 360))
            cv2.rectangle(tile, (0, 0), (480, 32), (15, 15, 15), -1)
            cv2.putText(tile, f"{recording_id}  {actual:.2f}s", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            tiles.append(tile)
    finally:
        cap.release()
        if cached is not None:
            cached.close()
    if not tiles:
        raise ValueError("No QA frames decoded")
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)])
    suffix = "landmark_qa" if landmarks_path else "raw_qa"
    destination = root / f"results/pilot/{recording_id}_{suffix}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), sheet):
        raise ValueError("Failed to save QA sheet")
    print(destination)
    return destination
