"""Extract included pilot recordings using the fixed reference intervals."""
import csv
from pathlib import Path
from facial_motion.acquire import ROOT,read_manifest
from facial_motion.extract import extract_recording

if __name__=='__main__':
    rows = read_manifest()
    if not (ROOT/'data/references.csv').is_file():
        raise FileNotFoundError('Pilot reference intervals are local-only inputs; restore data/references.csv before extraction. See docs/reproduction.md.')
    with (ROOT/'data/references.csv').open(newline='') as stream:
        references={r['recording_id']:r for r in csv.DictReader(stream)}
    for row in rows:
        if row['include']!='true':continue
        ref=references[row['recording_id']]
        extract_recording(row['recording_id'],reference=(float(ref['start_s']),float(ref['end_s'])))
        print('Extracted',row['recording_id'])
