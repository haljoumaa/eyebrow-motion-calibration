"""Public OSF inventory and checksum-verified, local-only downloads."""

import csv
import hashlib
import json
import re
import ssl
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import certifi

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = _SOURCE_ROOT if (_SOURCE_ROOT / "config.yaml").exists() else Path.cwd()
MANIFEST_FIELDS = [
    "recording_id", "source_url", "original_filename", "sha256", "participant_id",
    "distance", "head_movement", "direction", "eyebrow_condition", "fps", "width",
    "height", "duration_s", "duplicate_group", "reference_start_s", "reference_end_s",
    "include", "exclusion_reason", "source_folder", "source_file_id", "size_bytes",
    "local_path", "downloaded", "hash_verified", "condition_verified", "notes",
]


def fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "facial-motion-calibration/0.1"})
    with urllib.request.urlopen(request, context=ssl.create_default_context(cafile=certifi.where()), timeout=60) as r:
        return json.load(r)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination, expected_hash=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = sha256(destination)
        if expected_hash and actual != expected_hash:
            raise ValueError(f"Existing file checksum mismatch: {destination}")
        return actual
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "facial-motion-calibration/0.1"})
    try:
        with urllib.request.urlopen(request, context=ssl.create_default_context(cafile=certifi.where()), timeout=120) as r:
            with temporary.open("wb") as f:
                for chunk in iter(lambda: r.read(1024 * 1024), b""):
                    f.write(chunk)
        actual = sha256(temporary)
        if expected_hash and actual != expected_hash:
            raise ValueError(f"Downloaded checksum mismatch: {destination}")
        temporary.replace(destination)
        return actual
    finally:
        temporary.unlink(missing_ok=True)


def paged_files(url):
    while url:
        response = fetch_json(url)
        yield from response["data"]
        url = response.get("links", {}).get("next")


def provisional_labels(name):
    tokens = Path(name).stem.lower().split()
    distance = next((x for x in tokens if x in {"close", "middle", "medium", "far"}), "unknown")
    distance = "middle" if distance == "medium" else distance
    movement = next((x for x in tokens if x in {"pitch", "picth", "yaw", "roll"}), "unknown")
    movement = "pitch" if movement == "picth" else movement
    direction = next((x for x in tokens if x in {"up", "down", "left", "right"}), "unknown")
    if "back" in tokens:
        direction = "unknown"  # Visual verification required.
    state = "alternating" if "ernoer" in tokens else "neutral" if "noer" in tokens else "raised" if "er" in tokens else "unknown"
    return distance, movement, direction, state


def read_manifest(root=ROOT):
    path = Path(root) / "data/manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "Pilot manifest is a local-only input: " + str(path) +
            ". Restore the original reviewed inputs, or generate a provisional OSF "
            "inventory and review it as described in docs/reproduction.md."
        )
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(rows, root=ROOT):
    with (Path(root) / "data/manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def inventory(root=ROOT):
    root = Path(root)
    node = fetch_json("https://api.osf.io/v2/nodes/jvc42/")
    roots = list(paged_files("https://api.osf.io/v2/nodes/jvc42/files/osfstorage/?page%5Bsize%5D=100"))
    videos = next((x for x in roots if x["attributes"]["name"] == "video recordings head movement and eyebrows"), None)
    if videos is None:
        raise ValueError("Expected OSF video folder missing; inspect the current listing.")
    listing = []

    def walk(folder):
        url = folder["relationships"]["files"]["links"]["related"]["href"]
        for entry in paged_files(url):
            if entry["attributes"]["kind"] == "folder":
                walk(entry)
            elif entry["attributes"]["name"].lower().endswith(".mp4"):
                listing.append(entry)

    walk(videos)
    cache = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "node": node, "videos": listing}
    (root / "data/osf_inventory.json").write_text(json.dumps(cache, indent=2))
    previous = {x["recording_id"]: x for x in read_manifest(root)} if (root / "data/manifest.csv").exists() else {}
    rows = []
    counts = Counter(x["attributes"]["extra"]["hashes"].get("sha256") for x in listing)
    for entry in sorted(listing, key=lambda x: x["attributes"]["materialized_path"]):
        a = entry["attributes"]
        if Path(a["name"]).name != a["name"] or not re.fullmatch(r"[a-zA-Z0-9]+", entry["id"]):
            raise ValueError("Unexpected source filename or ID")
        rid = "osf_" + (a.get("guid") or entry["id"])
        distance, movement, direction, state = provisional_labels(a["name"])
        if state == "unknown" and "different head rotations neutral eyebrow position" in a["materialized_path"]:
            state = "neutral"
        checksum = a["extra"]["hashes"].get("sha256") or ""
        row = dict.fromkeys(MANIFEST_FIELDS, "")
        row.update(recording_id=rid, source_url=entry["links"]["download"], original_filename=a["name"],
                   sha256=checksum, participant_id="P01", distance=distance, head_movement=movement,
                   direction=direction, eyebrow_condition=state, duplicate_group="sha256_" + checksum if checksum else rid,
                   include="false", exclusion_reason="pending_visual_review", source_folder=str(Path(a["materialized_path"]).parent),
                   source_file_id=entry["id"], size_bytes=a["size"], local_path=f"data/raw/{rid}/{a['name']}",
                   downloaded="false", hash_verified="false", condition_verified="false",
                   notes="Filename labels are provisional; duplicate groups initially use source-reported hashes.")
        if rid in previous:
            old = previous[rid]
            if old["sha256"] != checksum:
                raise ValueError(f"Source content changed for {rid}; audit before refreshing the manifest")
            # Preserve recorded decisions and local acquisition metadata across inventory refreshes.
            for key in ["fps", "width", "height", "duration_s",
                        "reference_start_s", "reference_end_s", "include", "exclusion_reason", "duplicate_group",
                        "local_path", "downloaded", "hash_verified", "condition_verified", "notes"]:
                row[key] = old[key]
            if old["condition_verified"] == "true":
                for key in ["distance", "head_movement", "direction", "eyebrow_condition"]:
                    row[key] = old[key]
        rows.append(row)
    write_manifest(rows, root)
    print(f"OSF source entries: {len(rows)}; source-reported unique hashes: {len(counts)}")
    print(f"Declared node license: {node['data']['attributes']['node_license']}")
    for labels, n in sorted(Counter((x["distance"], x["head_movement"], x["eyebrow_condition"]) for x in rows).items()):
        print(" / ".join(labels), n)
    return rows


def acquire_recordings(ids, root=ROOT):
    root = Path(root)
    rows = read_manifest(root)
    available = {r["recording_id"] for r in rows}
    if set(ids) - available:
        raise ValueError(f"Unknown recording IDs: {set(ids) - available}")
    import cv2
    for row in rows:
        if row["recording_id"] not in ids:
            continue
        path = root / row["local_path"]
        digest = download(row["source_url"], path, row["sha256"] or None)
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot decode {path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            row.update(sha256=digest, downloaded="true", hash_verified="true", fps=fps,
                       width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                       duration_s=count / fps if fps > 0 else "")
        finally:
            cap.release()
        write_manifest(rows, root)
        print(f"Acquired {row['recording_id']}: {row['original_filename']} ({row['duration_s']:.2f}s)")
