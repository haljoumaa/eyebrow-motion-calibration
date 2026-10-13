"""Retrieve checksum-pinned upstream numerical inputs into a fresh directory."""
import argparse
import hashlib
import json
from pathlib import Path
import ssl
import urllib.request

from paths import REPO, fresh_output


def run(repo, output):
    repo = Path(repo).resolve()
    output = fresh_output(output, repo)
    manifest_path = repo / "research/upstream-assets.json"
    manifest = json.loads(manifest_path.read_text())
    records = []
    for asset in manifest["assets"]:
        target = output / asset["local_path"]
        if not target.resolve().is_relative_to(output):
            raise ValueError("Unsafe upstream input path")
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(asset["url"], headers={"User-Agent": "nonmanual-analysis/1.0"})
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=60, context=context) as response:
            contents = response.read()
        actual = hashlib.sha256(contents).hexdigest()
        if actual != asset["sha256"] or len(contents) != asset["bytes"]:
            raise ValueError(f"Pinned upstream input mismatch: {asset['local_path']}")
        target.write_bytes(contents)
        records.append(dict(path=asset["local_path"], sha256=actual, bytes=len(contents), url=asset["url"]))
    target = output / "research/upstream-assets.json"
    target.write_bytes(manifest_path.read_bytes())
    (output / "retrieval.json").write_text(json.dumps(dict(passed=True, inputs=records), indent=2) + "\n")
    print(f"PASS: {len(records)} pinned upstream inputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--output", type=Path, required=True, help="Fresh input root; existing paths are rejected")
    args = parser.parse_args()
    run(args.repo, args.output)
