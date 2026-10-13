"""Recheck the 2024 paper's Tables 1-2 from released intermediate deviations."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from facial_motion.acquire import sha256
from verification_output import fresh_output

ROOT=Path(__file__).resolve().parents[1]
def verify(root=ROOT, output=None):
    root=Path(root).resolve();out=fresh_output(output,root)
    asset=json.loads((root/'research/published-table-assets.json').read_text())['assets'][0]
    path=root/asset['local_path'];assert sha256(path)==asset['sha256']
    frame=pd.read_csv(path)
    inner=[c for c in frame if c.startswith('inner') and c.endswith('lower')]
    outer=[c for c in frame if c.startswith('outer') and not c.startswith('outer_18')]
    assert len(inner)==len(outer)==35
    frame['MPH_inner']=frame[inner].mean(axis=1);frame['MPH_outer']=frame[outer].mean(axis=1)
    frame['OF_inner']=frame[['inner_20_23_perp_dist36_45_3d','inner_20_23_perp_dist39_42_3d']].mean(axis=1)
    frame['OF_outer']=frame[['outer_18_25_perp_dist36_45_3d','outer_18_25_perp_dist39_42_3d']].mean(axis=1)
    means=frame.groupby(frame.columns[0])[['MPH_inner','MPH_outer','OF_inner','OF_outer']].mean()
    saved=pd.read_csv(root/'results/literature/reproduction_2024_tables.csv')
    values=[float(means.loc[r.recording+(' noER' if r.condition=='neutral' else ' ER'),r.measure]) for r in saved.itertuples()]
    np.testing.assert_allclose(values,saved.reproduced,rtol=1e-12,atol=1e-12)
    differences=np.abs(np.asarray(values)-saved.published);assert len(values)==48 and (differences<=.0005+1e-12).all()
    record={'passed':True,'matched_published_entries':48,'maximum_difference_from_published_rounded_values':float(differences.max()),'maximum_difference_from_saved_reproduction':float(np.abs(np.asarray(values)-saved.reproduced).max()),'scope':'Table arithmetic from released intermediate deviations; no raw extraction or significance-test reproduction','input_sha256':asset['sha256']}
    if out is not None:
        with (out/'published-table-check.json').open('x') as stream:
            stream.write(json.dumps(record,indent=2)+'\n')
    print('PASS: 48 published entries and saved table arithmetic')
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,help='Optional fresh directory; default: stdout only')
    args=parser.parse_args();verify(args.root,args.output)
