"""Raw-frame review and conservative screening for re-encoded overlapping footage.

Similarity is a review aid, never an automatic claim of independent recordings.
All face-image outputs remain local under results/audit/.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .acquire import ROOT, read_manifest


def face_crop(image, distance):
    """Fixed crops by camera condition; preserve source aspect ratio."""
    h, w = image.shape[:2]
    bounds = {"close": (.28, .08, .74, .94),
              "middle": (.36, .18, .68, .91),
              "far": (.40, .22, .64, .89)}
    x0, y0, x1, y1 = bounds[distance]
    return image[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]


def raw_sheet(row, times, destination, root=ROOT, points=None, config=None, columns=6):
    """Save raw or marked frames with actual decoder timestamps and undistorted crops."""
    cap = cv2.VideoCapture(str(Path(root) / row["local_path"]))
    tiles = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t)*1000)
            ok, frame = cap.read()
            if not ok:
                continue
            actual = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if points is not None:
                from .extract import draw_landmarks
                idx = int(np.argmin(np.abs(points["timestamps"]-actual)))
                frame = draw_landmarks(frame, points["landmarks"][idx], config, actual)
            crop = face_crop(frame, row["distance"])
            h, w = crop.shape[:2]
            factor = min(320/w, 300/h)
            resized = cv2.resize(crop, (int(w*factor), int(h*factor)))
            tile = np.full((336, 320, 3), 20, np.uint8)
            yy, xx = 36+(300-resized.shape[0])//2, (320-resized.shape[1])//2
            tile[yy:yy+resized.shape[0], xx:xx+resized.shape[1]] = resized
            cv2.putText(tile, f"{row['distance']}  t={actual:.2f}s", (8, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .60, (255,255,255), 1)
            tiles.append(tile)
    finally:
        cap.release()
    if not tiles:
        raise ValueError("No review frames decoded")
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i+columns]) for i in range(0,len(tiles),columns)])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), sheet):
        raise ValueError("Unable to write review sheet")
    return destination


def review_all(root=ROOT):
    root = Path(root)
    for row in read_manifest(root):
        if row["head_movement"] != "pitch" or row["downloaded"] != "true":
            continue
        rid, duration = row["recording_id"], float(row["duration_s"])
        out = root / "results/audit" / rid
        raw_sheet(row, np.linspace(0, duration-.08, 12), out / "overview.jpg", root)
        raw_sheet(row, np.arange(0, min(1.44,duration), .12), out / "start.jpg", root)
        if row["eyebrow_condition"] == "alternating":
            for page, start in enumerate(np.arange(0,duration,4.8)):
                raw_sheet(row, np.arange(start,min(start+4.8,duration-.01),.20),
                          out / f"states_{page+1}.jpg", root)
        print(rid, row["original_filename"], flush=True)


def consecutive_runs(mask):
    edges = np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return list(zip(np.flatnonzero(edges==1), np.flatnonzero(edges==-1)))


def overlap_features(row, root=ROOT):
    cap = cv2.VideoCapture(str(Path(root)/row["local_path"]))
    frames, timestamps = [], []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(face_crop(frame,row["distance"]),cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(gray,(48,64)).astype(np.float32))
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
    finally:
        cap.release()
    return np.asarray(frames),np.asarray(timestamps)


def aligned_overlap(a,b,minimum_frames=25,threshold=3.0):
    """Return the most similar aligned contiguous window and strict low-error runs.

    Compare every temporal offset for equal-rate clips. A >=1s run below 3 gray
    levels suggests a re-encoding match, subject to raw review. This can miss
    changed crops, speed or colour; absence is not proof of independence.
    """
    best = None
    longest = 0
    for offset in range(-len(b)+minimum_frames,len(a)-minimum_frames+1):
        index_a,bi=max(0,offset),max(0,-offset)
        count=min(len(a)-index_a,len(b)-bi)
        if count < minimum_frames:
            continue
        errors=np.abs(a[index_a:index_a+count]-b[bi:bi+count]).mean(axis=(1,2))
        windows=np.convolve(errors,np.ones(minimum_frames)/minimum_frames,mode="valid")
        k=int(np.argmin(windows))
        candidate=(float(windows[k]),index_a+k,bi+k)
        if best is None or candidate[0]<best[0]:
            best=candidate
        longest=max(longest,max((end-start for start,end in consecutive_runs(errors<threshold)),default=0))
    if best is None:
        raise ValueError("Clips shorter than overlap screening window")
    return {"best_window_mae_gray":best[0],"frame_a":best[1],"frame_b":best[2],
            "longest_low_error_run_frames":longest,"minimum_window_frames":minimum_frames,
            "low_error_threshold_gray":threshold}


def screen_neutral_overlap(root=ROOT):
    root=Path(root)
    rows=[r for r in read_manifest(root) if r["head_movement"]=="pitch" and
          r["eyebrow_condition"]=="neutral" and r["downloaded"]=="true"]
    features={r["recording_id"]:overlap_features(r,root) for r in rows}
    results=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if a["distance"]!=b["distance"]:
                continue
            if not np.isclose(float(a["fps"]),float(b["fps"])):
                raise ValueError("Unequal rates need explicit overlap review")
            fa,ta=features[a["recording_id"]]
            fb,tb=features[b["recording_id"]]
            result=aligned_overlap(fa,fb,minimum_frames=int(round(float(a["fps"]))))
            result.update(recording_a=a["recording_id"],recording_b=b["recording_id"],
                          distance=a["distance"],time_a_s=float(ta[result["frame_a"]]),
                          time_b_s=float(tb[result["frame_b"]]))
            results.append(result)
    table=pd.DataFrame(results)
    out=root/"results/audit/neutral_overlap_screen.csv"
    out.parent.mkdir(parents=True,exist_ok=True)
    table.to_csv(out,index=False)
    print(table[["recording_a","recording_b","best_window_mae_gray","longest_low_error_run_frames"]].to_string(index=False))
    return table


def review_geometry(root=ROOT):
    root=Path(root)
    from .extract import load_config
    rows={r["recording_id"]:r for r in read_manifest(root)}
    for q in pd.read_csv(root/"data/qa_frames.csv").to_dict("records"):
        rid=q["recording_id"]
        times=[q[k] for k in ["reference_time_s","mid_tilt_time_s","max_tilt_time_s"]]
        with np.load(root/f"data/cache/{rid}_landmarks.npz") as raw:
            raw_sheet(rows[rid],times,root/f"results/audit/{rid}/geometry.jpg",root,
                      points=raw,config=load_config(root/"config.yaml"),columns=3)
        print(rid,flush=True)
