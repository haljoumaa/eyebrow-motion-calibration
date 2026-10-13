"""Retrieve fixed external inputs without changing reviewed manifests or labels."""
import argparse
import json
from pathlib import Path
from facial_motion.acquire import download, read_manifest

ROOT = Path(__file__).resolve().parents[1]

def fetch(group, root=ROOT):
    root = Path(root)
    items = []
    if group in {'openface', 'all'}:
        for item in json.loads((root / 'research/upstream-assets.json').read_text())['assets']:
            items.append((item['url'], item['local_path'], item['sha256']))
    if group == 'literature':
        for item in json.loads((root / 'research/published-table-assets.json').read_text())['assets']:
            items.append((item['url'], item['local_path'], item['sha256']))
    if group in {'pilot', 'all'}:
        pilot_rows = read_manifest(root)  # Check local inputs before any download.
        model = json.loads((root / 'models/model_metadata.json').read_text())
        items.append((model['url'], 'models/face_landmarker.task', model['sha256']))
        items.extend((r['source_url'], r['local_path'], r['sha256']) for r in pilot_rows
                     if r['head_movement'] == 'pitch')
    for url, path, checksum in items:
        download(url, root / path, checksum)
        print('Verified', path)
    return len(items)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['openface', 'pilot', 'literature', 'all'], default='openface')
    args = parser.parse_args()
    fetch(args.group)
