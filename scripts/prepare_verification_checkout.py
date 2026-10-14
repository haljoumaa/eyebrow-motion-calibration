"""Prepare a verification copy with restored source observations.

The copy combines saved predictions with checksum-verified external inputs and
checks the restored CSVs against their original hashes without fitting models.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from facial_motion.openface_data import load_neutral_data, TARGETS


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink():
        raise ValueError('Unsafe artifact path: ' + relative)
    if not path.resolve().is_relative_to(root):
        raise ValueError('Artifact resolves outside checkout: ' + relative)
    return path


def prepare(output, inputs=ROOT, pilot_cache=None):
    inputs, output = Path(inputs).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output must be a fresh directory: ' + str(output))
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'tmp'):
        raise ValueError('Inside the candidate, use an ignored tmp/ output')
    if ROOT.is_relative_to(output) or inputs.is_relative_to(output):
        raise ValueError('Output may not contain source checkout or input root')
    if output.is_relative_to(inputs / 'research/upstream-review'):
        raise ValueError('Output overlaps external assets')
    projections = json.loads((ROOT / 'research/prediction-projections.json').read_text())
    rows = load_neutral_data(inputs).rows
    neutral_bytes = rows.to_csv(index=False).encode()
    lex = list(csv.DictReader(io.StringIO(neutral_bytes.decode())))
    generated = {'results/openface/recording-split/neutral_rows.csv': neutral_bytes}
    for record in projections:
        public = safe(ROOT, record['public_path']).read_bytes()
        if sha(public) != record['public_sha256']:
            raise ValueError('Saved prediction projection changed')
        reader = csv.DictReader(io.StringIO(public.decode()))
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=record['original_columns'], lineterminator='\n')
        writer.writeheader(); count = 0
        for row in reader:
            source = lex[int(row['neutral_row'])]
            for identity in ['video_name', 'frame', 'speaker', 'sentence']:
                if row[identity] != source[identity]:
                    raise ValueError('Prediction/source identity differs: ' + identity)
            for i, label in enumerate(['inner', 'outer']):
                row['observed_' + label] = source[TARGETS[i]]
                if 'corrected_' + label in record['original_columns']:
                    row['corrected_' + label] = str(float(source[TARGETS[i]]) - float(row['predicted_' + label]))
            writer.writerow(row); count += 1
        data = buffer.getvalue().encode()
        if count != record['rows'] or sha(data) != record['original_sha256']:
            raise ValueError('Restored prediction CSV differs from the original. Use the recorded runtime.')
        generated[record['original_path']] = data

    metadata = json.loads((ROOT / 'results/openface/recording-split/run_metadata.json').read_text())
    if sha(neutral_bytes) != metadata['artifact_sha256']['neutral_rows.csv']:
        raise ValueError('Restored neutral subset differs from the original')
    manifest = json.loads((ROOT / 'research/upstream-assets.json').read_text())
    external = []
    for asset in manifest['assets']:
        data = safe(inputs, asset['local_path']).read_bytes()
        if sha(data) != asset['sha256']:
            raise ValueError('External input mismatch: ' + asset['local_path'])
        external.append((asset['local_path'], data))
    cached = []
    if pilot_cache:
        pilot_cache = Path(pilot_cache).resolve()
        if output.is_relative_to(pilot_cache) or pilot_cache.is_relative_to(output):
            raise ValueError('Output overlaps pilot cache')
        metadata_path = inputs / 'results/face-landmarker/run_metadata.json'
        if not metadata_path.is_file():
            raise FileNotFoundError('Pilot cache verification requires the original run metadata under --inputs. See docs/reproduction.md.')
        metadata = json.loads(metadata_path.read_text())
        for relative, expected in metadata['input_hashes'].items():
            if not relative.startswith('data/cache/'):
                continue
            data = (pilot_cache / Path(relative).name).read_bytes()
            if sha(data) != expected:
                raise ValueError('Pilot measurement cache mismatch: ' + relative)
            cached.append((relative, data))
        for relative in ['data/manifest.csv', 'data/references.csv', 'data/annotations.csv',
                         'data/qa_frames.csv', 'data/audit_review.json',
                         'results/face-landmarker/predictions.csv',
                         'results/face-landmarker/recording_metrics.csv',
                         'results/face-landmarker/run_metadata.json']:
            data = safe(inputs, relative).read_bytes()
            if relative in metadata['input_hashes'] and sha(data) != metadata['input_hashes'][relative]:
                raise ValueError('Pilot input mismatch: ' + relative)
            cached.append((relative, data))
    excluded_names = {'.git', '.venv', 'tmp', 'build', 'dist', '.pytest_cache', '__pycache__'}
    excluded_paths = {'research/upstream-review', 'data/raw', 'data/cache', 'results/audit',
                      'results/pilot', 'results/repeats', 'results/demo'}
    def ignore(directory, names):
        current = Path(directory)
        ignored = []
        for name in names:
            path = current / name
            if (name in excluded_names or name.endswith('.egg-info') or
                path.relative_to(ROOT).as_posix() in excluded_paths or
                path.relative_to(ROOT).as_posix() == 'models/face_landmarker.task'):
                ignored.append(name)
            elif path.is_symlink():
                raise ValueError('Source symlink is not allowed: ' + str(path))
        return ignored
    shutil.copytree(ROOT, output, ignore=ignore)
    for relative, data in list(generated.items()) + external + cached:
        target = safe(output, relative); target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    record = {'passed': True, 'private_scientific_copy': str(output),
              'source_csvs_restored_exactly': len(generated), 'saved_prediction_rows': sum(p['rows'] for p in projections),
              'checksum_verified_external_assets': len(external), 'checksum_verified_pilot_measurement_caches': len(cached),
              'models_fitted': 0, 'predictions_recomputed': 0, 'tolerances_weakened': False}
    print(json.dumps(record, indent=2))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=ROOT)
    parser.add_argument('--pilot-cache', type=Path)
    args = parser.parse_args()
    prepare(args.output, args.inputs, args.pilot_cache)
