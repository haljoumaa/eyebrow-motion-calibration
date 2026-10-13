"""Reference-relative stability and within-recording visible-state separation."""

import numpy as np
from sklearn.metrics import roc_auc_score


def stability(values, reference_height, mask=None):
    values = np.asarray(values, float)
    usable = np.isfinite(values)
    if mask is not None:
        usable &= np.asarray(mask, bool)
    error = values[usable] - reference_height
    return {"n": int(usable.sum()), "mae": float(np.abs(error).mean()) if len(error) else np.nan,
            "rmse": float(np.sqrt((error**2).mean())) if len(error) else np.nan}


def state_labels(timestamps, intervals):
    timestamps = np.asarray(timestamps, float)
    labels = np.full(len(timestamps), np.nan)
    claimed = np.zeros(len(timestamps), bool)
    for interval in intervals:
        start, end = float(interval["start_s"]), float(interval["end_s"])
        if end <= start or interval["state"] not in {"neutral", "raised"}:
            raise ValueError("Invalid manual state interval")
        mask = (timestamps >= start) & (timestamps < end)
        if (claimed & mask).any():
            raise ValueError("Manual state intervals overlap")
        claimed |= mask
        if str(interval["uncertain"]).lower() == "true":
            continue
        labels[mask] = float(interval["state"] == "raised")
    return labels


def preservation_auc(values, labels, mask=None):
    values, labels = np.asarray(values, float), np.asarray(labels, float)
    usable = np.isfinite(values) & np.isfinite(labels)
    if mask is not None:
        usable &= np.asarray(mask, bool)
    counts = {"n_neutral": int((usable & (labels == 0)).sum()),
              "n_raised": int((usable & (labels == 1)).sum())}
    if np.unique(labels[usable]).size != 2:
        return {"n": int(usable.sum()), "auc": np.nan, "reason": "both_states_required", **counts}
    return {"n": int(usable.sum()), "auc": float(roc_auc_score(labels[usable], values[usable])), "reason": "", **counts}
