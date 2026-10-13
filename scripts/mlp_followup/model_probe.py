"""Descriptive identity-interaction probes at fixed, valid training poses.

These are model response checks, not estimates of anatomical or causal effects.
"""
import argparse
import json
from pathlib import Path
from paths import REPO, fresh_output
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def predict(weights, prep, numeric, signers, sentences):
    identifiers = prep.get('identities', ['speaker','sentence'])
    blocks = [numeric]
    for category in identifiers:
        values = signers if category == 'speaker' else sentences
        blocks.append(np.column_stack([values == v for v in prep['categories'][category]]).astype(float))
    z = (np.column_stack(blocks)-np.array(prep['scaler_mean']))/np.array(prep['scaler_scale'])
    if weights['model']=='mlp':
        hidden = np.maximum(0, z@np.array(weights['coefs'][0])+np.array(weights['intercepts'][0]))
        return hidden@np.array(weights['coefs'][1])+np.array(weights['intercepts'][1])
    return z@np.array(weights['coefs']).T+np.array(weights['intercepts'])


def run(repo, results, output):
    repo, results = Path(repo).resolve(), Path(results).resolve()
    out = fresh_output(output, repo, results)
    base=repo/'results/openface/recording-split'
    rows=pd.read_csv(base/'neutral_rows.csv')
    allocation=pd.read_csv(repo/'configs/recording-folds.csv')
    original=json.loads((base/'model_runs.json').read_text())
    prep=json.loads((base/'preprocessing.json').read_text())
    new=json.loads((results/'fit_runs.json').read_text())
    newprep=json.loads((results/'preprocessing.json').read_text())
    result=[]; reference_rows=[]
    for fold in range(4):
        test=rows[rows.video_name.isin(allocation.loc[allocation.validation_fold.eq(fold),'video_name'])]
        train=rows[~rows.video_name.isin(test.video_name)]
        # Five actual training poses nearest the robustly scaled training median.
        pose=train[[f'pose_{k}{a}' for k in ['R','T'] for a in 'xyz']]
        scaled=RobustScaler().fit_transform(pose)
        selected=np.argsort(np.sum(scaled**2,axis=1),kind='stable')[:5]
        references=train.iloc[selected]
        reference_rows.extend(dict(fold=fold,neutral_row=int(r)) for r in references.neutral_row)
        cats=prep[str(fold)]['full']['categories']
        grid=[(s,t) for s in cats['speaker'] for t in cats['sentence']]
        signers=np.array([p[0] for p in grid]); sentences=np.array([p[1] for p in grid])
        trainpairs=set(map(tuple,train[['speaker','sentence']].to_numpy()))
        testpairs=set(map(tuple,test[['speaker','sentence']].to_numpy()))
        seen=np.array([p in trainpairs for p in grid]); held=np.array([p in testpairs for p in grid])
        models=[(r,prep[str(fold)]['full'],'original') for r in original if r['fold']==fold and r['model'] in ['mlp','ridge']]
        models.extend((r,newprep[f'outer{fold}_{r["arm"]}'],'followup_'+r['arm']) for r in new if r['phase']=='outer' and r['fold']==fold)
        for weights,state,label in models:
            probe=[]
            for _, pose_row in references.iterrows():
                numeric=np.tile([v for a in 'xyz' for v in [pose_row['pose_R'+a],np.cos(pose_row['pose_R'+a]),pose_row['pose_T'+a]]],(len(grid),1))
                predicted=predict(weights,state,numeric,signers,sentences).reshape(len(cats['speaker']),len(cats['sentence']),2)
                additive=predicted.mean(0,keepdims=True)+predicted.mean(1,keepdims=True)-predicted.mean((0,1),keepdims=True)
                interaction=(predicted-additive).reshape(-1,2)
                probe.append(dict(interaction_rms=float(np.sqrt((interaction**2).mean())),
                    seen_pair_mean_interaction=float(interaction[seen].mean()),
                    held_pair_mean_interaction=float(interaction[held].mean()),
                    held_pair_mean_prediction=float(predicted.reshape(-1,2)[held].mean())))
            result.append(dict(fold=fold, label=label, model=weights['model'], seed=weights.get('seed'),
                               **{k: np.mean([p[k] for p in probe]) for k in probe[0]}))
    pd.DataFrame(result).to_csv(out/'identity_interaction_probe.csv',index=False)
    (out/'identity_interaction_probe_method.json').write_text(json.dumps(dict(
        reference_rows=reference_rows, method='For each outer fold, use five actual training poses nearest the robustly scaled training median. Cross known training signer and sentence categories at each fixed pose. Remove row and column means to obtain the two-way additive residual. Average probe summaries over five poses.',
        interpretation='Synthetic identity swaps may not be physically meaningful or covered. This checks model dependence on indicator combinations; it does not estimate a causal identity effect or new evaluation performance.'),indent=2)+'\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--results", type=Path, help="Fitted-result directory; default: retained exploratory results")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory; existing paths are rejected")
    args = parser.parse_args()
    results = args.results or args.repo / "results/openface/mlp-followup"
    run(args.repo, results, args.output)
