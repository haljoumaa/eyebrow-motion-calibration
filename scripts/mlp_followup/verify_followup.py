"""Independent saved-weight, split, preprocessing and selection audit of the follow-up."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from paths import REPO, fresh_output
sys.path.insert(0, str(REPO / "scripts"))
from protocol_hashes import verify_source_hashes
sys.path.pop(0)
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_implementation(repo, base, plan):
    """Check fitted-plan provenance and the active reproduction implementation."""
    manifest = json.loads((repo/'research/mlp-provenance.json').read_text())
    try:
        verify_source_hashes(repo, manifest['fixed_inputs_sha256'])
    except ValueError as error:
        raise AssertionError(str(error)) from error
    assert digest(repo/'configs/recording-folds.csv') == plan['source']['folds_sha256']
    assert plan['exploratory'] is True
    active = repo/'scripts/mlp_followup/followup.py'
    actual = digest(active)
    assert actual == manifest['active_fit_script_sha256']
    assert plan['source']['script_sha256'] in {actual, manifest['prefit_script_sha256']}
    try:
        verify_source_hashes(repo, manifest['implementation_sha256'])
    except ValueError as error:
        raise AssertionError(str(error)) from error
    retained_checked = 0
    if base == repo/'results/openface/mlp-followup':
        for relative, expected in manifest['result_sha256'].items():
            assert digest(repo/relative) == expected, relative
            retained_checked += 1
    return dict(prefit_script_sha256=plan['source']['script_sha256'],
                active_script_sha256=actual, retained_files_checked=retained_checked)


def run(repo, results, output=None):
    repo, base = Path(repo).resolve(), Path(results).resolve()
    out = fresh_output(output, repo, base) if output is not None else None
    rows=pd.read_csv(repo/'results/openface/recording-split/neutral_rows.csv').set_index('neutral_row',drop=False)
    allocation=pd.read_csv(repo/'configs/recording-folds.csv')
    pred=pd.read_csv(base/'predictions.csv')
    runs=json.loads((base/'fit_runs.json').read_text())
    states=json.loads((base/'preprocessing.json').read_text())
    inner_assign=pd.read_csv(base/'inner_assignments.csv')
    inner_scores=pd.read_csv(base/'inner_recording_scores.csv')
    selections=pd.read_csv(base/'selections.csv')
    saved_rec=pd.read_csv(base/'recording_scores.csv')
    plan=json.loads((base/'plan.json').read_text())
    provenance = verify_implementation(repo, base, plan)
    assert (len(rows), rows.video_name.nunique(), rows.speaker.nunique()) == (4414, 63, 8)
    assert len(pred) == 176560 and len(states) == 52
    assert len(runs)==184 and len([r for r in runs if r['phase']=='outer'])==40
    max_prediction_error=0.; max_metric_error=0.
    for key,state in states.items():
        fold=int(key.split('_')[0].removeprefix('outer'))
        held=set(allocation.loc[allocation.validation_fold.eq(fold),'video_name'])
        training=rows.loc[state['training_rows']]
        assert set(training.video_name).isdisjoint(held)
        if '_inner' in key:
            inner_fold=int(key.split('_')[1].removeprefix('inner'))
            assignment=inner_assign[inner_assign.outer_fold.eq(fold)&inner_assign.inner_fold.eq(inner_fold)]
            intended=set(assignment.loc[assignment.partition.eq('training'),'video_name'])
            valid=set(assignment.loc[assignment.partition.eq('validation'),'video_name'])
            assert intended.isdisjoint(valid) and (intended|valid)==set(rows.video_name)-held
            assert set(training.video_name)==intended
            assert set(rows[rows.video_name.isin(valid)].speaker)<=set(training.speaker)
        else:
            assert set(training.video_name)==set(rows.video_name)-held
        np.testing.assert_array_equal(training.neutral_row, rows.loc[rows.video_name.isin(training.video_name), 'neutral_row'])
        blocks=[v for axis in 'xyz' for v in [training['pose_R'+axis].to_numpy(),np.cos(training['pose_R'+axis].to_numpy()),training['pose_T'+axis].to_numpy()]]
        for cat in state['identities']:
            assert state['categories'][cat]==sorted(training[cat].unique())
            blocks.extend(training[cat].eq(v).to_numpy(dtype=float) for v in state['categories'][cat])
        design=np.column_stack(blocks)
        np.testing.assert_allclose(design.mean(0),state['scaler_mean'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(design.var(0),state['scaler_var'],rtol=1e-12,atol=1e-12)
        assert len(training)==state['n_training_frames']
    with threadpool_limits(limits=1):
        for weights in [r for r in runs if r['phase']=='outer']:
            fold,arm,seed=weights['fold'],weights['arm'],weights['seed']
            state=states[f'outer{fold}_{arm}']
            current=pred[pred.fold.eq(fold)&pred.arm.eq(arm)&(pred.seed.isna() if seed is None else pred.seed.eq(seed))]
            assert len(current)==4414 and not current.neutral_row.duplicated().any()
            held=set(allocation.loc[allocation.validation_fold.eq(fold),'video_name'])
            for partition in ['training','validation']:
                part=current[current.partition.eq(partition)]
                expected=rows[rows.video_name.isin(held) if partition=='validation' else ~rows.video_name.isin(held)]
                np.testing.assert_array_equal(part.neutral_row,expected.neutral_row)
                for col in ['video_name','frame','speaker','sentence','confidence','success']:
                    np.testing.assert_array_equal(part[col],expected[col])
                np.testing.assert_allclose(part[['observed_inner','observed_outer']],expected[['inner_perp_dist39_42_3d','outer_perp_dist39_42_3d']],rtol=1e-12,atol=1e-12)
            source=rows.loc[current.neutral_row]
            blocks=[v for axis in 'xyz' for v in [source['pose_R'+axis].to_numpy(),np.cos(source['pose_R'+axis].to_numpy()),source['pose_T'+axis].to_numpy()]]
            for cat in state['identities']:
                blocks.extend(source[cat].eq(v).to_numpy(dtype=float) for v in state['categories'][cat])
            z=(np.column_stack(blocks)-np.array(state['scaler_mean']))/np.array(state['scaler_scale'])
            if weights['model']=='mlp':
                h=np.maximum(0,z@np.array(weights['coefs'][0])+np.array(weights['intercepts'][0]))
                rebuilt=h@np.array(weights['coefs'][1])+np.array(weights['intercepts'][1])
            else:
                rebuilt=z@np.array(weights['coefs']).T+np.array(weights['intercepts'])
            actual=current[['predicted_inner','predicted_outer']].to_numpy()
            max_prediction_error=max(max_prediction_error,float(np.abs(rebuilt-actual).max()))
            np.testing.assert_allclose(rebuilt,actual,rtol=1e-12,atol=1e-12)
            errors=(current[['observed_inner','observed_outer']].to_numpy()-actual)**2
            calc=current[['video_name','partition']].copy()
            calc['inner_mse']=errors[:,0]; calc['outer_mse']=errors[:,1]; calc['joint_mse']=errors.mean(1)
            calc=calc.groupby(['video_name','partition'])[['inner_mse','outer_mse','joint_mse']].mean()
            saved=saved_rec[saved_rec.fold.eq(fold)&saved_rec.arm.eq(arm)&(saved_rec.seed.isna() if seed is None else saved_rec.seed.eq(seed))].set_index(['video_name','partition'])
            for col in ['inner_mse','outer_mse','joint_mse']:
                np.testing.assert_allclose(calc[col],saved.loc[calc.index,col],rtol=1e-12,atol=1e-12)
                max_metric_error=max(max_metric_error,float(np.abs(calc[col]-saved.loc[calc.index,col]).max()))
    for selection in selections[selections.arm.ne('selected_policy')].to_dict('records'):
        part=inner_scores[inner_scores.outer_fold.eq(selection['fold'])&inner_scores.arm.eq(selection['arm'])]
        assert not part.duplicated(['alpha','video_name']).any()
        means=part.groupby('alpha').joint_mse.mean(); complete=part.groupby('alpha').complete.all()
        chosen=sorted(complete.index[complete],key=lambda alpha:(means[alpha],-alpha))[0]
        assert chosen==selection['alpha']
        np.testing.assert_allclose(means[chosen],selection['inner_equal_recording_joint_mse'],rtol=1e-12,atol=1e-12)
    for selection in selections[selections.arm.eq('selected_policy')].to_dict('records'):
        competitors=selections[selections.fold.eq(selection['fold'])&selections.arm.isin(['full','no_sentence','pose_only'])]
        assert competitors.sort_values('inner_equal_recording_joint_mse').arm.iloc[0]==selection['selected_arm']
    result=dict(passed=True,prediction_rows_checked=len(pred),outer_models_checked=40,
        training_only_preprocessors_checked=len(states),inner_group_assignments_checked=12,
        settings_selected_from_grouped_training_validation_only=True,
        max_absolute_prediction_difference=max_prediction_error,max_absolute_metric_difference=max_metric_error,
        incomplete_inner_fits=sum(r['phase']=='inner' and not r['complete'] for r in runs),
        incomplete_outer_fits=sum(r['phase']=='outer' and not r['complete'] for r in runs),
        implementation_provenance=provenance,tolerances=dict(rtol=1e-12,atol=1e-12))
    if out is not None:
        (out/'followup_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--results", type=Path, help="Fitted-result directory; default: retained exploratory results")
    parser.add_argument("--output", type=Path, help="Optional fresh verification directory; omitted means stdout only")
    args = parser.parse_args()
    results = args.results or args.repo / "results/openface/mlp-followup"
    print(json.dumps(run(args.repo, results, args.output), indent=2))
