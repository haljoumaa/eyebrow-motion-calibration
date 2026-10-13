"""Bounded exploratory nested recording validation of identity ablations and L2 strength."""
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
from paths import REPO, fresh_output
import platform
import sys
import time
import warnings
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits, threadpool_info

SEEDS = [42, 7, 19]
ARMS = {'full': ['speaker', 'sentence'], 'no_sentence': ['speaker'], 'pose_only': []}
MLP_ALPHAS = [.0001, 1., 100.]
RIDGE_ALPHAS = [.001, 1., 100.]
TARGETS = ['inner_perp_dist39_42_3d', 'outer_perp_dist39_42_3d']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class Preparation:
    def __init__(self, identities):
        self.identities = identities

    @staticmethod
    def numeric(rows):
        return np.column_stack([v for axis in 'xyz' for v in
            [rows['pose_R' + axis].to_numpy(), np.cos(rows['pose_R' + axis].to_numpy()), rows['pose_T' + axis].to_numpy()]])

    def fit(self, rows):
        numeric = self.numeric(rows)
        if self.identities:
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            design = np.column_stack([numeric, self.encoder.fit_transform(rows[self.identities])])
        else:
            design = numeric
        self.scaler = StandardScaler().fit(design)
        self.training_rows = rows.neutral_row.tolist()
        np.testing.assert_allclose(self.scaler.mean_, design.mean(0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(self.scaler.var_, design.var(0), rtol=1e-12, atol=1e-12)
        return self

    def transform(self, rows):
        numeric = self.numeric(rows)
        if self.identities:
            if 'speaker' in self.identities:
                assert set(rows.speaker) <= set(self.encoder.categories_[self.identities.index('speaker')])
            numeric = np.column_stack([numeric, self.encoder.transform(rows[self.identities])])
        return self.scaler.transform(numeric)

    def state(self):
        return dict(identities=self.identities, training_rows=self.training_rows,
            categories={k: v.tolist() for k, v in zip(self.identities, self.encoder.categories_)} if self.identities else {},
            scaler_mean=self.scaler.mean_.tolist(), scaler_scale=self.scaler.scale_.tolist(),
            scaler_var=self.scaler.var_.tolist(), n_training_frames=int(self.scaler.n_samples_seen_),
            target_scaling=False)


def score(part, predicted):
    squared = (part[TARGETS].to_numpy() - predicted)**2
    frame = pd.DataFrame(dict(video_name=part.video_name.to_numpy(), speaker=part.speaker.to_numpy(),
        sentence=part.sentence.to_numpy(), inner_mse=squared[:, 0], outer_mse=squared[:, 1], joint_mse=squared.mean(1)))
    g = frame.groupby('video_name', sort=False)
    result = g[['inner_mse','outer_mse','joint_mse']].mean()
    result['speaker'] = g.speaker.first(); result['sentence'] = g.sentence.first(); result['n_frames'] = g.size()
    return result.reset_index()


def estimator(model, alpha, seed):
    if model == 'ridge':
        return Ridge(alpha=alpha)
    return MLPRegressor(hidden_layer_sizes=(40,), activation='relu', solver='adam', alpha=alpha,
                        max_iter=1500, early_stopping=False, random_state=seed)


def fit(model, alpha, seed, x, y):
    est = estimator(model, alpha, seed)
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        est.fit(x, y)
    complete = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return est, dict(complete=complete, seconds=time.monotonic()-started,
                     n_iter=int(est.n_iter_) if model == 'mlp' else None,
                     warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught])


def run(repo, inputs, output):
    repo, inputs = Path(repo).resolve(), Path(inputs).resolve()
    output = fresh_output(output, repo, inputs)
    sys.path.insert(0, str(repo / 'src'))
    from facial_motion.openface_data import load_neutral_data, sha256
    from facial_motion.openface_recording import recording_splits
    data = load_neutral_data(inputs)
    rows = data.rows
    allocation = pd.read_csv(repo / 'configs/recording-folds.csv')
    assert sha256(repo / 'configs/recording-folds.csv') == 'f888c92b5c2a96fc6cc74ebddbdf01895bf00f3197f37037cd91954df58afed0'
    outer = recording_splits(rows, allocation)
    plan = dict(recorded_before_fitting_oslo=datetime.now(ZoneInfo('Europe/Oslo')).isoformat(),
        exploratory=True, rationale='All signer-sentence pairs are unseen; pose novelty and confidence weakly associated with original MLP excess. Test identity interactions and L2 shrinkage, retaining original architecture and optimizer.',
        mlp_arms=ARMS, mlp_alphas=MLP_ALPHAS, ridge_alphas=RIDGE_ALPHAS,
        architecture='40 ReLU units / Adam / max_iter=1500 / early_stopping=False / unscaled targets',
        primary_seed=42, sensitivity_seeds=[7, 19], inner_splits='GroupKFold(n_splits=3, shuffle=False), groups=video_name, source row order preserved',
        selection='Per arm and outer fold, minimize pooled equal-recording inner out-of-fold joint MSE at seed 42. Ties prefer stronger alpha. No selection using outer results or sensitivity seeds.',
        candidate_count='9 MLP settings plus 3 full-input ridge settings; 108 inner MLP fits, 36 inner ridge fits, 36 outer MLP fits, 4 outer ridge fits',
        convergence='Fixed 1500-iteration budget; all warnings retained. Incomplete settings excluded from selection; no extra trials. If none complete in an arm, fail.',
        training_weighting='equal frames', evaluation_weighting='equal recordings pooled across folds; frame-weighted and equal-signer checks reported separately',
        exclusions=0, preprocessing='Fit categories and complete design standardization separately on each inner/outer training set; no targets in preprocessing',
        source=dict(repo=str(repo), folds_sha256=sha256(repo / 'configs/recording-folds.csv'),
                    script_sha256=sha256(Path(__file__)), input_sha256=data.provenance['input_sha256']),
        python=platform.python_version(), platform=platform.platform(),
        packages={p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','threadpoolctl','joblib']})
    write(output / 'plan.json', plan)
    assignments, validation_scores, runs, predictions, recording_scores, states, selections = [], [], [], [], [], {}, []
    with threadpool_limits(limits=1):
        write(output / 'threadpools.json', threadpool_info())
        for fold, (train_idx, test_idx) in enumerate(outer):
            train, test = rows.iloc[train_idx], rows.iloc[test_idx]
            inner = list(GroupKFold(n_splits=3).split(train, groups=train.video_name))
            for inner_fold, (inner_train, inner_test) in enumerate(inner):
                a, b = train.iloc[inner_train], train.iloc[inner_test]
                assert set(a.video_name).isdisjoint(b.video_name)
                assert set(a.speaker) >= set(b.speaker)
                assert set(a.video_name | b.video_name) if False else True
                for partition, part in [('training',a), ('validation',b)]:
                    for video, g in part.groupby('video_name', sort=False):
                        assignments.append(dict(outer_fold=fold, inner_fold=inner_fold, partition=partition,
                                                video_name=video, speaker=g.speaker.iloc[0], n_frames=len(g)))
                for arm, identities in ARMS.items():
                    prep = Preparation(identities).fit(a)
                    x, xv = prep.transform(a), prep.transform(b.drop(columns=TARGETS))
                    states[f'outer{fold}_inner{inner_fold}_{arm}'] = prep.state()
                    for alpha in MLP_ALPHAS:
                        fit_id = f'outer{fold}_inner{inner_fold}_{arm}_alpha{alpha}'
                        est, status = fit('mlp', alpha, 42, x, a[TARGETS].to_numpy())
                        table = score(b, est.predict(xv)); table['outer_fold'] = fold; table['inner_fold'] = inner_fold
                        table['arm'] = arm; table['alpha'] = alpha; table['complete'] = status['complete']
                        validation_scores.append(table)
                        runs.append(dict(fit_id=fit_id, model='mlp', phase='inner', fold=fold, arm=arm, alpha=alpha, seed=42, **status))
                        print(f'{fit_id}: iterations={status["n_iter"]}, complete={status["complete"]}', flush=True)
                    if arm == 'full':
                        for alpha in RIDGE_ALPHAS:
                            est, status = fit('ridge', alpha, None, x, a[TARGETS].to_numpy())
                            table = score(b, est.predict(xv)); table['outer_fold'] = fold; table['inner_fold'] = inner_fold
                            table['arm'] = 'ridge'; table['alpha'] = alpha; table['complete'] = status['complete']
                            validation_scores.append(table)
                            runs.append(dict(fit_id=f'outer{fold}_inner{inner_fold}_ridge_alpha{alpha}', model='ridge', phase='inner', fold=fold, arm='ridge', alpha=alpha, seed=None, **status))
            inner_scores = pd.concat(validation_scores, ignore_index=True)
            inner_current = inner_scores[inner_scores.outer_fold.eq(fold)]
            choices = {}
            for arm in [*ARMS, 'ridge']:
                arm_scores = inner_current[inner_current.arm.eq(arm)]
                eligible = arm_scores.groupby('alpha').complete.all()
                means = arm_scores.groupby('alpha').joint_mse.mean()
                assert len(arm_scores) == train.video_name.nunique()*3
                eligible_alphas = eligible.index[eligible]
                if not len(eligible_alphas):
                    raise RuntimeError(f'No complete candidate for fold {fold}, arm {arm}')
                chosen = sorted(eligible_alphas, key=lambda alpha: (means[alpha], -alpha))[0]
                choices[arm] = chosen
                selections.append(dict(fold=fold, arm=arm, alpha=float(chosen), inner_equal_recording_joint_mse=float(means[chosen]),
                                       n_inner_validation_recordings=train.video_name.nunique()))
            # This overall choice, too, uses only inner scores; arm priority resolves exact ties.
            selected_arm = min(ARMS, key=lambda arm: next(s['inner_equal_recording_joint_mse'] for s in selections if s['fold']==fold and s['arm']==arm))
            selections.append(dict(fold=fold, arm='selected_policy', selected_arm=selected_arm, alpha=float(choices[selected_arm]),
                inner_equal_recording_joint_mse=next(s['inner_equal_recording_joint_mse'] for s in selections if s['fold']==fold and s['arm']==selected_arm)))
            for arm in [*ARMS, 'ridge']:
                prep = Preparation(ARMS.get(arm, ARMS['full'])).fit(train)
                x, xt = prep.transform(train), prep.transform(test.drop(columns=TARGETS))
                states[f'outer{fold}_{arm}'] = prep.state()
                model = 'ridge' if arm == 'ridge' else 'mlp'
                for seed in [None] if model == 'ridge' else SEEDS:
                    est, status = fit(model, choices[arm], seed, x, train[TARGETS].to_numpy())
                    fit_id = f'outer{fold}_{arm}_seed{seed}_alpha{choices[arm]}'
                    parameters = est.get_params()
                    runs.append(dict(fit_id=fit_id, model=model, phase='outer', fold=fold, arm=arm, alpha=float(choices[arm]), seed=seed,
                        parameters=parameters, coefs=[v.tolist() for v in est.coefs_] if model=='mlp' else est.coef_.tolist(),
                        intercepts=[v.tolist() for v in est.intercepts_] if model=='mlp' else est.intercept_.tolist(), **status))
                    for partition, part, matrix in [('training',train,x), ('validation',test,xt)]:
                        predicted = est.predict(matrix)
                        table = score(part, predicted); table['fold']=fold; table['arm']=arm; table['seed']=seed
                        table['alpha']=choices[arm]; table['partition']=partition; table['complete']=status['complete']
                        recording_scores.append(table)
                        # Save every prediction for reproducible arithmetic (no pickle or executable model load).
                        p = part[['neutral_row','video_name','frame','speaker','sentence','confidence','success']].copy()
                        p['fold']=fold; p['arm']=arm; p['seed']=seed; p['partition']=partition
                        for j, label in enumerate(['inner','outer']):
                            p['observed_'+label]=part[TARGETS[j]].to_numpy(); p['predicted_'+label]=predicted[:,j]
                        predictions.append(p)
                    print(f'{fit_id}: complete={status["complete"]}; selected arm={selected_arm}', flush=True)
            # Save after each outer fold for transparent progress and interruption recovery.
            pd.concat(validation_scores, ignore_index=True).to_csv(output / 'inner_recording_scores.csv', index=False)
            pd.DataFrame(assignments).to_csv(output / 'inner_assignments.csv', index=False)
            pd.DataFrame(selections).to_csv(output / 'selections.csv', index=False)
            pd.concat(recording_scores, ignore_index=True).to_csv(output / 'recording_scores.csv', index=False)
            pd.concat(predictions, ignore_index=True).to_csv(output / 'predictions.csv', index=False)
            write(output / 'fit_runs.json', runs); write(output / 'preprocessing.json', states)
    write(output / 'completion.json', dict(completed_oslo=datetime.now(ZoneInfo('Europe/Oslo')).isoformat(),
        n_inner_mlp_fits=sum(r['phase']=='inner' and r['model']=='mlp' for r in runs),
        n_outer_mlp_fits=sum(r['phase']=='outer' and r['model']=='mlp' for r in runs),
        warning_fits=sum(bool(r['warnings']) for r in runs), incomplete_fits=sum(not r['complete'] for r in runs)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--inputs", type=Path, required=True, help="Root containing research/upstream-assets.json and the pinned inputs")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory; existing paths are rejected")
    args = parser.parse_args()
    run(args.repo, args.inputs, args.output)
