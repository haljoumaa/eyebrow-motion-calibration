"""Recording-level errors, training-only pose coverage, identity and quality diagnostics."""
import argparse
import json
from pathlib import Path
from paths import REPO, fresh_output
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, RobustScaler
from threadpoolctl import threadpool_limits

SEEDS = [42, 7, 19]
POSE = ['pose_Rx', 'pose_Ry', 'pose_Rz', 'pose_Tx', 'pose_Ty', 'pose_Tz']


def correlation(x, y, fold=None, controls=None):
    """Descriptive rank associations; no frame-independent significance claims."""
    x, y = np.asarray(x), np.asarray(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    if fold is None and controls is None:
        return float(spearmanr(x, y).statistic)
    design = np.ones((len(x), 1))
    if fold is not None:
        design = np.column_stack([design, pd.get_dummies(fold, drop_first=True).to_numpy(dtype=float)])
    if controls is not None:
        design = np.column_stack([design, *[rankdata(v) for v in controls]])
    residuals = [rankdata(v) - design @ np.linalg.lstsq(design, rankdata(v), rcond=None)[0] for v in [x, y]]
    if min(np.std(v) for v in residuals) < 1e-12:
        return None
    return float(np.corrcoef(*residuals)[0, 1])


def coverage(train, test, scope, fold):
    """Ranges and Euclidean nearest-neighbour distances across all six raw pose inputs.

    Scales are fitted to training frames. The novelty threshold is the 95th
    percentile of training distances to a different recording, avoiding a
    within-recording nearest-neighbour reference. Two scale choices are retained.
    """
    train_x, test_x = train[POSE].to_numpy(), test[POSE].to_numpy()
    low, high = train_x.min(0), train_x.max(0)
    outside = (test_x < low) | (test_x > high)
    result = pd.DataFrame({'neutral_row': test.neutral_row.to_numpy(),
                           'fold': fold, 'scope': scope,
                           'outside_any_range': outside.any(1)})
    ranges = []
    for j, feature in enumerate(POSE):
        result['outside_' + feature] = outside[:, j]
        ranges.append(dict(fold=fold, scope=scope, feature=feature,
                           train_min=low[j], train_max=high[j],
                           train_q01=np.quantile(train_x[:, j], .01), train_q99=np.quantile(train_x[:, j], .99),
                           validation_min=test_x[:, j].min(), validation_max=test_x[:, j].max(),
                           validation_outside_fraction=outside[:, j].mean(),
                           n_training_frames=len(train), n_validation_frames=len(test)))
    for name, scaler in [('standard', StandardScaler()), ('robust', RobustScaler())]:
        scaled = scaler.fit_transform(train_x)
        query = scaler.transform(test_x)
        # Exact distances, with nearest training neighbours never restricted by outcome.
        nn = NearestNeighbors(n_neighbors=1).fit(scaled)
        distance = nn.kneighbors(query, return_distance=True)[0][:, 0] / np.sqrt(6)
        reference = np.empty(len(train))
        video = train.video_name.to_numpy()
        for recording in train.video_name.unique():
            mask = video == recording
            neighbour = NearestNeighbors(n_neighbors=1).fit(scaled[~mask])
            reference[mask] = neighbour.kneighbors(scaled[mask])[0][:, 0] / np.sqrt(6)
        threshold = float(np.quantile(reference, .95))
        result['nn_' + name] = distance
        result['novel_' + name] = distance > threshold
        result['threshold_' + name] = threshold
    return result, ranges


def run(repo, inputs, output):
    repo, inputs = Path(repo).resolve(), Path(inputs).resolve()
    out = fresh_output(output, repo, inputs)
    sys.path.insert(0, str(repo / 'src'))
    from facial_motion.openface_data import load_neutral_data
    rows = load_neutral_data(inputs).rows
    base = repo / 'results/openface/recording-split'
    recs = pd.read_csv(base / 'recording_metrics.csv')
    held = recs[recs.partition.eq('validation') & recs.model.isin(['ridge', 'mlp'])].copy()
    held['model_label'] = np.where(held.model.eq('mlp'), 'MLP seed ' + held.seed.fillna(0).astype(int).astype(str), 'Ridge')
    held.to_csv(out / 'recording_errors.csv', index=False)
    metrics = []
    for label, group in held.groupby('model_label', sort=False):
        training = recs[recs.partition.eq('training') & recs.model.eq(group.model.iloc[0])]
        if group.model.iloc[0] == 'mlp':
            training = training[training.seed.eq(group.seed.iloc[0])]
        metrics.append(dict(model=label, train_equal_recording_joint_mse=training.joint_mse.mean(),
            equal_recording_joint_mse=group.joint_mse.mean(), median_recording_joint_mse=group.joint_mse.median(),
            inner_mse=group.inner_mse.mean(), outer_mse=group.outer_mse.mean(),
            inner_mae=group.inner_mae.mean(), outer_mae=group.outer_mae.mean(),
            frame_weighted_joint_mse=np.average(group.joint_mse, weights=group.n_frames),
            equal_signer_joint_mse=group.groupby('speaker').joint_mse.mean().mean()))
    pd.DataFrame(metrics).to_csv(out / 'original_model_summary.csv', index=False)
    pieces, ranges, identity = [], [], []
    with threadpool_limits(limits=1):
        for fold in range(4):
            videos = set(held.loc[held.fold.eq(fold), 'video_name'])
            train, test = rows[~rows.video_name.isin(videos)], rows[rows.video_name.isin(videos)]
            part, limits = coverage(train, test, 'all_signers', fold)
            pieces.append(part); ranges.extend(limits)
            for signer, signer_test in test.groupby('speaker', sort=False):
                part, limits = coverage(train[train.speaker.eq(signer)], signer_test, 'same_signer', fold)
                part['speaker'] = signer
                for limit in limits:
                    limit['speaker'] = signer
                pieces.append(part); ranges.extend(limits)
            for video, group in test.groupby('video_name', sort=False):
                signer, sentence = group[['speaker', 'sentence']].iloc[0]
                pair = train[train.speaker.eq(signer) & train.sentence.eq(sentence)]
                identity.append(dict(fold=fold, video_name=video, speaker=signer, sentence=sentence,
                    pair_seen=bool(len(pair)), pair_training_recordings=pair.video_name.nunique(),
                    signer_seen=signer in set(train.speaker), sentence_seen=sentence in set(train.sentence),
                    signer_training_recordings=train.loc[train.speaker.eq(signer), 'video_name'].nunique(),
                    sentence_training_recordings=train.loc[train.sentence.eq(sentence), 'video_name'].nunique(),
                    n_frames=len(group), mean_confidence=group.confidence.mean(),
                    low_confidence_fraction=group.quality_low_confidence.mean(),
                    failed_tracking_fraction=group.success.eq(0).mean()))
            print(f'Coverage fold {fold} complete', flush=True)
    frame = pd.concat(pieces, ignore_index=True)
    frame.to_csv(out / 'pose_coverage_frames.csv', index=False)
    pd.DataFrame(ranges).to_csv(out / 'pose_feature_ranges.csv', index=False)
    identity = pd.DataFrame(identity)
    identity.to_csv(out / 'identity_coverage.csv', index=False)
    diagnostics = identity.copy()
    for scope in ['all_signers', 'same_signer']:
        cov = frame[frame.scope.eq(scope)].merge(rows[['neutral_row', 'video_name']], on='neutral_row', validate='many_to_one')
        aggregates = cov.groupby('video_name').agg(
            range_fraction=('outside_any_range', 'mean'),
            nn_robust_median=('nn_robust', 'median'), nn_robust_mean=('nn_robust', 'mean'),
            nn_robust_p95=('nn_robust', lambda x: np.quantile(x, .95)),
            nn_standard_median=('nn_standard', 'median'),
            novel_robust_fraction=('novel_robust', 'mean'), novel_standard_fraction=('novel_standard', 'mean'))
        diagnostics = diagnostics.merge(aggregates.add_prefix(scope + '_'), on='video_name', validate='one_to_one')
    ridge = held[held.model.eq('ridge')].set_index('video_name')
    for col in ['joint_mse', 'inner_mse', 'outer_mse']:
        diagnostics['ridge_' + col] = diagnostics.video_name.map(ridge[col])
    failures = []
    for seed in SEEDS:
        m = held[held.model.eq('mlp') & held.seed.eq(seed)].set_index('video_name')
        for col in ['joint_mse', 'inner_mse', 'outer_mse']:
            diagnostics[f'mlp{seed}_' + col] = diagnostics.video_name.map(m[col])
            diagnostics[f'excess{seed}_' + col] = diagnostics[f'mlp{seed}_' + col] - diagnostics['ridge_' + col]
        diagnostics[f'mlp{seed}_ridge_ratio'] = diagnostics[f'mlp{seed}_joint_mse'] / diagnostics.ridge_joint_mse
        diff = diagnostics[f'excess{seed}_joint_mse']
        positive = diff.clip(lower=0)
        order = diagnostics.sort_values(f'excess{seed}_joint_mse', ascending=False)
        failures.append(dict(seed=seed, worse_recordings=int((diff > 0).sum()),
            worse_inner=int((diagnostics[f'excess{seed}_inner_mse'] > 0).sum()),
            worse_outer=int((diagnostics[f'excess{seed}_outer_mse'] > 0).sum()),
            signers_with_positive_mean_excess=int(diagnostics.groupby('speaker')[f'excess{seed}_joint_mse'].mean().gt(0).sum()),
            sentences_with_positive_mean_excess=int(diagnostics.groupby('sentence')[f'excess{seed}_joint_mse'].mean().gt(0).sum()),
            top1_share_positive_excess=float(positive.max() / positive.sum()),
            top5_share_positive_excess=float(positive.nlargest(5).sum() / positive.sum()),
            top10_share_positive_excess=float(positive.nlargest(10).sum() / positive.sum()),
            mean_excess_excluding_top5=float(order.iloc[5:][f'excess{seed}_joint_mse'].mean()),
            top_recording=order.video_name.iloc[0]))
    diagnostics.to_csv(out / 'recording_diagnostics.csv', index=False)
    pd.DataFrame(failures).to_csv(out / 'failure_extent.csv', index=False)
    for key in ['speaker', 'sentence', 'fold']:
        columns = [c for c in diagnostics if c.startswith(('ridge_', 'mlp', 'excess')) and 'ratio' not in c]
        table = diagnostics.groupby(key)[columns].mean()
        table.insert(0, 'n_recordings', diagnostics.groupby(key).size())
        table.to_csv(out / f'errors_by_{key}.csv')
    associations = []
    predictors = ['all_signers_range_fraction', 'same_signer_range_fraction',
        'all_signers_nn_robust_median', 'same_signer_nn_robust_median',
        'all_signers_nn_standard_median', 'same_signer_nn_standard_median',
        'all_signers_novel_robust_fraction', 'same_signer_novel_robust_fraction',
        'n_frames', 'mean_confidence', 'low_confidence_fraction', 'signer_training_recordings', 'sentence_training_recordings']
    for feature in predictors:
        for outcome in ['ridge_joint_mse'] + [f'{prefix}{s}_{target}_mse' for prefix in ['mlp', 'excess'] for s in SEEDS for target in ['joint', 'inner', 'outer']]:
            associations.append(dict(feature=feature, outcome=outcome, n_recordings=63,
                spearman=correlation(diagnostics[feature], diagnostics[outcome]),
                rank_correlation_adjusted_fold=correlation(diagnostics[feature], diagnostics[outcome], diagnostics.fold),
                rank_correlation_adjusted_fold_frames_confidence=correlation(diagnostics[feature], diagnostics[outcome],
                    diagnostics.fold, [diagnostics.n_frames, diagnostics.mean_confidence])))
    pd.DataFrame(associations).to_csv(out / 'associations.csv', index=False)
    # Quality and signed-bias sensitivity: primary scoring always retains every frame.
    pred = pd.read_csv(base / 'predictions.csv')
    pred = pred[pred.partition.eq('validation') & pred.model.isin(['ridge', 'mlp'])].copy()
    for label in ['inner', 'outer']:
        pred['residual_' + label] = pred['observed_' + label] - pred['predicted_' + label]
    pred['joint_squared_error'] = (pred.residual_inner ** 2 + pred.residual_outer ** 2) / 2
    sensitivity, bias = [], []
    for (model, seed), group in pred.groupby(['model', 'seed'], dropna=False, sort=False):
        for name, mask in [('all_frames', np.ones(len(group), bool)), ('confidence_ge_0.8', group.confidence.ge(.8))]:
            g = group[mask].groupby('video_name').joint_squared_error.mean()
            sensitivity.append(dict(model=model, seed=seed, subset=name, n_recordings=len(g), n_frames=int(np.sum(mask)),
                                    equal_recording_joint_mse=g.mean()))
        for video, g in group.groupby('video_name'):
            bias.append(dict(model=model, seed=seed, video_name=video,
                inner_mean_residual=g.residual_inner.mean(), outer_mean_residual=g.residual_outer.mean(),
                joint_bias_squared=(g.residual_inner.mean()**2 + g.residual_outer.mean()**2)/2,
                joint_mse=g.joint_squared_error.mean()))
    pd.DataFrame(sensitivity).to_csv(out / 'quality_sensitivity.csv', index=False)
    pd.DataFrame(bias).to_csv(out / 'recording_bias.csv', index=False)
    summary = dict(n_frames=len(rows), n_recordings=rows.video_name.nunique(), n_signers=rows.speaker.nunique(),
        n_sentences=rows.sentence.nunique(), n_distinct_pairs=rows[['speaker','sentence']].drop_duplicates().shape[0],
        pair_seen_recordings=int(identity.pair_seen.sum()), signer_unseen_recordings=int((~identity.signer_seen).sum()),
        sentence_unseen_recordings=int((~identity.sentence_seen).sum()), low_confidence_frames=int(rows.quality_low_confidence.sum()),
        standard_translation_scale_note='Training-only standard deviations can be dominated by released extreme translations; robust-scale NN is also reported.',
        largest_absolute_translations={c: float(rows[c].abs().max()) for c in POSE[3:]},
        failures=failures, original_metrics=metrics)
    (out / 'diagnostic_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--inputs", type=Path, required=True, help="Root containing research/upstream-assets.json and the pinned inputs")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory; existing paths are rejected")
    args = parser.parse_args()
    run(args.repo, args.inputs, args.output)
