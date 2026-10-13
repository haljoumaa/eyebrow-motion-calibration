"""Rerun the fixed OpenFace experiments and local pilot, then compare saved results."""
import argparse,json
from pathlib import Path
from facial_motion.openface_data import ROOT
from facial_motion.openface_baselines import run_baselines
from facial_motion.openface_reproduction import run_reproduction
from facial_motion.openface_recording import run_recording_evaluation
from facial_motion.evaluate import run_evaluation
from facial_motion.reproducibility import compare_table,compare_values,verify_fresh_process
from protocol_hashes import verify_source_hashes,compare_source_hashes


def repeat(output, root=ROOT):
    root,output=Path(root).resolve(),Path(output).resolve()
    for experiment in ("baselines","frame-split","recording-split"):
        metadata=json.loads((root/f"results/openface/{experiment}/run_metadata.json").read_text())
        verify_source_hashes(root,metadata["source_sha256"])
    required = ['data/manifest.csv', 'data/references.csv', 'data/annotations.csv',
                'results/face-landmarker/predictions.csv',
                'results/face-landmarker/recording_metrics.csv',
                'results/face-landmarker/run_metadata.json']
    missing = [p for p in required if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError('Combined replay requires local-only pilot artifacts before fitting: ' +
                                ', '.join(missing) + '. See docs/reproduction.md.')
    output.mkdir(parents=True,exist_ok=False)
    run_baselines(output/'baselines',root)
    run_reproduction(output/'frame-split',root)
    run_recording_evaluation(output/'recording-split',root,save_bundles=True)
    run_evaluation(root,output/'face-landmarker')
    table_checks=[];state_checks={};protocol_revisions={}
    for original,repeated in [(root/'results/openface/baselines',output/'baselines'),(root/'results/openface/frame-split',output/'frame-split'),(root/'results/openface/recording-split',output/'recording-split'),(root/'results/face-landmarker',output/'face-landmarker')]:
        for path in sorted(original.glob('*.csv')):
            if path.name == 'predictions-public.csv':
                continue
            table_checks.append({'experiment':original.name,**compare_table(path,repeated/path.name)})
        for name in ['preprocessing.json','training_means.json','unknown_sentence_coverage.json','convergence_remedies.json']:
            if (original/name).exists():state_checks[original.name+'/'+name]=compare_values(json.loads((original/name).read_text()),json.loads((repeated/name).read_text()))
        if (original/'model_runs.json').exists():
            def stable(p):return [{k:v for k,v in r.items() if k not in {'fit_seconds','completed_at_oslo'}} for r in json.loads(p.read_text())]
            state_checks[original.name+'/model_runs.json']=compare_values(stable(original/'model_runs.json'),stable(repeated/'model_runs.json'))
        first,last=[json.loads((p/'run_metadata.json').read_text()) for p in [original,repeated]]
        for key in ['config','folds','inputs','input_hashes','source_sha256','source_hashes','packages','python']:
            if key in first:
                if key in {"source_sha256","source_hashes","input_hashes"}:
                    protocol_revisions[original.name]=compare_source_hashes(root,first[key],last[key])
                else:
                    state_checks[original.name+'/metadata.'+key]=compare_values(first[key],last[key])
    verify_fresh_process(root,output,replay=output/'recording-split')
    record={'passed':True,'source_hash_checks_with_recorded_revisions':protocol_revisions,'rtol':1e-12,'atol':1e-12,'tables':table_checks,'state_max_absolute_differences':state_checks,'maximum_table_difference':max(c['max_absolute_numeric_difference'] for c in table_checks),'fit_times_and_completion_timestamps_excluded':True,'numerical_equivalence_is_environment_specific':True}
    (output/'replay-check.json').write_text(json.dumps(record,indent=2)+'\n')
    print('PASS: fixed experiments, state comparisons and 28 fresh-process bundle reloads')
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='Fresh directory; existing paths are rejected')
    repeat(parser.parse_args().output)
