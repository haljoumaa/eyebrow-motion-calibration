"""Summarize fixed and exploratory OpenFace results and plot recording diagnostics."""
import argparse
import json
import os
from pathlib import Path
from paths import REPO, fresh_output
import numpy as np
import pandas as pd

LABELS={'full':'MLP (pose + signer + sentence), grouped α',
        'no_sentence':'MLP (pose + signer), grouped α',
        'pose_only':'MLP (pose only), grouped α',
        'ridge':'Ridge (pose + signer + sentence), grouped α'}
COLORS={42:'#2369a3',7:'#d27a25',19:'#7551a2'}


def run(repo, results, diagnostics, probe, output):
    repo, results, diagnostics, probe = [Path(p).resolve() for p in (repo, results, diagnostics, probe)]
    fresh = fresh_output(output, repo, results, diagnostics, probe)
    out = fresh / 'diagnostics'; out.mkdir()
    figures = fresh / 'figures'; figures.mkdir()
    os.environ['MPLCONFIGDIR']=str(fresh/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=pd.read_csv(diagnostics/'recording_diagnostics.csv')
    original=pd.read_csv(diagnostics/'original_model_summary.csv')
    new=pd.read_csv(results/'recording_scores.csv')
    selection=pd.read_csv(results/'selections.csv')
    probe=pd.read_csv(probe/'identity_interaction_probe.csv')
    summary=[]; comparisons=[]
    for r in original.to_dict('records'):
        seed=None if r['model']=='Ridge' else int(r['model'].split()[-1])
        summary.append(dict(model='Fixed ridge' if seed is None else f'Fixed MLP, seed {seed}',
            status='original fixed experiment',primary_seed=seed,
            training_joint_mse=r['train_equal_recording_joint_mse'],joint_mse=r['equal_recording_joint_mse'],
            inner_mse=r['inner_mse'],outer_mse=r['outer_mse'],
            sensitivity_joint_mse_min=None,sensitivity_joint_mse_max=None,
            frame_weighted_joint_mse=r['frame_weighted_joint_mse'],equal_signer_joint_mse=r['equal_signer_joint_mse']))
    for arm in ['full','no_sentence','pose_only','ridge']:
        part=new[new.arm.eq(arm)]
        held=part[part.partition.eq('validation')]
        seed_scores=held.groupby('seed',dropna=False).joint_mse.mean()
        primary=held[held.seed.isna() if arm=='ridge' else held.seed.eq(42)]
        training=part[part.partition.eq('training')&(part.seed.isna() if arm=='ridge' else part.seed.eq(42))]
        summary.append(dict(model=LABELS[arm],status='exploratory grouped selection',primary_seed=None if arm=='ridge' else 42,
            training_joint_mse=training.joint_mse.mean(),joint_mse=primary.joint_mse.mean(),
            inner_mse=primary.inner_mse.mean(),outer_mse=primary.outer_mse.mean(),
            sensitivity_joint_mse_min=seed_scores.min() if arm!='ridge' else None,
            sensitivity_joint_mse_max=seed_scores.max() if arm!='ridge' else None,
            frame_weighted_joint_mse=np.average(primary.joint_mse,weights=primary.n_frames),
            equal_signer_joint_mse=primary.groupby('speaker').joint_mse.mean().mean()))
        for seed,g in held.groupby('seed',dropna=False):
            paired=g.merge(d[['video_name','ridge_joint_mse']],on='video_name',validate='one_to_one')
            assert len(paired)==63 and paired.n_frames.sum()==4414
            difference=paired.joint_mse-paired.ridge_joint_mse
            comparisons.append(dict(arm=arm,seed=None if pd.isna(seed) else int(seed),joint_mse=g.joint_mse.mean(),
                inner_mse=g.inner_mse.mean(),outer_mse=g.outer_mse.mean(),
                beats_fixed_ridge_recordings=int(difference.lt(0).sum()),mean_difference_fixed_ridge=difference.mean(),
                median_difference_fixed_ridge=difference.median(),all_fits_complete=bool(g.complete.all())))
    pd.DataFrame(summary).to_csv(out/'main_table.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'followup_comparisons.csv',index=False)
    findings=[
        dict(topic='Failure extent',finding='MLP loses to ridge on 58/60/57 of 63 recordings at seeds 42/7/19; all eight signers and all ten sentences have positive mean excess error.',kind='observation'),
        dict(topic='Pose coverage',finding='Same-signer robust pose novelty versus excess joint MSE: Spearman 0.095/0.147/0.165. Global range novelty association at seed 42: 0.020.',kind='association'),
        dict(topic='Identity',finding='Zero of 63 held-out signer-sentence pairs are in training; every signer and sentence is individually known. There is no seen-pair contrast.',kind='observation'),
        dict(topic='Offsets',finding='Squared recording-level bias explains 95.6%/97.5%/97.1% of equal-recording MLP joint MSE at seeds 42/7/19.',kind='observation'),
        dict(topic='Follow-up',finding='All folds select alpha=100 for each MLP arm. The selected MLP (pose + signer) excludes sentence identity. Its joint MSE is 2.623 (seed 42), range 2.597–2.623; fixed ridge is 2.471.',kind='exploratory observation'),
        dict(topic='Interpretation',finding='Recording offsets and the identity-response probe are consistent with overfitting to signer–sentence combinations. The diagnostics do not identify the cause; the influence of extreme translations was not isolated.',kind='descriptive interpretation')]
    pd.DataFrame(findings).to_csv(out/'main_findings.csv',index=False)
    # Matching frame weights for descriptive frame-split / recording-split comparison.
    historic=pd.read_csv(repo/'results/openface/frame-split/historical_summary.csv')
    matched=[]
    for row in historic[historic.partition.eq('validation')&historic.model.isin(['ridge','mlp'])].to_dict('records'):
        label='Ridge' if row['model']=='ridge' else f'MLP seed {int(row["seed"])}'
        matched.append(dict(model=label,historical_frame_weighted_joint_mse=row['joint_mse'],
            recording_frame_weighted_joint_mse=original.loc[original.model.eq(label),'frame_weighted_joint_mse'].iloc[0],
            note='Matching within-frame weighting; different validation populations and dependent historical splits. No causal degradation percentage.'))
    pd.DataFrame(matched).to_csv(out/'weight_matched_split_comparison.csv',index=False)
    # Additional direct description of released translation extremes and scaling.
    rows=pd.read_csv(repo/'results/openface/recording-split/neutral_rows.csv')
    extreme=rows[['pose_Tx','pose_Ty','pose_Tz']].abs().max(axis=1).gt(1e6)
    scale=[]
    for fold in range(4):
        train=rows[~rows.video_name.isin(d.loc[d.fold.eq(fold),'video_name'])]
        for col in ['pose_Tx','pose_Ty','pose_Tz']:
            iqr=train[col].quantile(.75)-train[col].quantile(.25)
            scale.append(dict(fold=fold,feature=col,training_sd=train[col].std(ddof=0),training_iqr=iqr,
                              sd_to_iqr_ratio=train[col].std(ddof=0)/iqr))
    pd.DataFrame(scale).to_csv(out/'translation_scale_diagnostics.csv',index=False)
    (out/'released_extreme_translations.json').write_text(json.dumps(dict(
        definition='Any absolute released translation greater than 1,000,000; descriptive flag, not an exclusion or claimed physical threshold',
        frames=int(extreme.sum()),low_confidence_frames=int(rows.loc[extreme,'quality_low_confidence'].sum()),
        recordings=rows.loc[extreme,'video_name'].unique().tolist()),indent=2)+'\n')
    # Show all 63 recordings and all three fixed-experiment seeds.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','axes.titlesize':12})
    fig,axs=plt.subplots(2,2,figsize=(14.5,10.5))
    ax=axs[0,0]
    ordered=d.sort_values(['speaker','sentence']).reset_index(drop=True)
    ordered.insert(0,'figure_order',np.arange(1,len(ordered)+1))
    ordered[['figure_order','video_name','speaker','sentence','fold']].to_csv(figures/'figure_recording_order.csv',index=False)
    for seed,offset in [(42,-.23),(7,0),(19,.23)]:
        for target,marker in [('inner','o'),('outer','x')]:
            ratio=ordered[f'mlp{seed}_{target}_mse']/ordered[f'ridge_{target}_mse']
            ax.scatter(ordered.figure_order+offset,ratio,s=16,color=COLORS[seed],marker=marker,alpha=.75,
                       label=f'{seed}, {target}')
    ax.axhline(1,color='#3f454c',lw=1)
    ax.set_yscale('log'); ax.set_ylim(.025,1000); ax.set_xlim(0,64)
    centres=ordered.groupby('speaker',sort=False).figure_order.mean()
    ax.set_xticks(centres.to_numpy(),centres.index)
    for start in ordered.groupby('speaker',sort=False).figure_order.min().iloc[1:]:
        ax.axvline(start-.5,color='#e3e5e7',lw=.8,zorder=0)
    ax.set_title('A  MLP-to-ridge error ratios by recording',loc='left')
    ax.set_ylabel('MLP / ridge MSE (log scale)'); ax.set_xlabel('All 63 recordings, grouped by signer')
    ax=axs[0,1]
    for seed in [42,7,19]:
        ax.scatter(d.same_signer_nn_robust_median,d[f'excess{seed}_joint_mse'],color=COLORS[seed],s=23,alpha=.65,label=f'Seed {seed}')
    ax.axhline(0,color='#3f454c',lw=1); ax.set_yscale('symlog',linthresh=1)
    ax.set_xlabel('Median distance to a training pose from the same signer\nSix pose features; training IQR scales; different recordings')
    ax.set_ylabel('MLP − ridge joint MSE'); ax.set_title('B  Pose novelty and excess error',loc='left')
    ax.text(.98,.97,'Spearman ρ: 0.095 / 0.147 / 0.165',transform=ax.transAxes,ha='right',va='top',fontsize=9)
    ax=axs[1,0]
    labels=['Fixed MLP','Full-input MLP\nα=100','MLP (pose + signer)\nα=100']; groups=['original','followup_full','followup_no_sentence']
    for j,(key,color,name) in enumerate([('seen_pair_mean_interaction','#6b8f9e','Pairs present in training'),('held_pair_mean_interaction','#cb765b','Held-out pairs')]):
        values=[]; lower=[]; upper=[]
        for group in groups:
            seed_values=probe[probe.label.eq(group)&probe.model.eq('mlp')].groupby('seed')[key].mean()
            values.append(seed_values.mean()); lower.append(seed_values.mean()-seed_values.min()); upper.append(seed_values.max()-seed_values.mean())
        ax.bar(np.arange(3)+(j-.5)*.32,values,width=.3,color=color,label=name,yerr=np.array([lower,upper]),capsize=3)
    ax.axhline(0,color='#3f454c',lw=1); ax.set_xticks(range(3),labels,fontsize=9)
    ax.set_ylabel('Mean deviation from additive identity response\n(released output units)')
    ax.set_title('C  Responses to signer–sentence combinations',loc='left')
    ax.legend(frameon=False,fontsize=8,loc='upper right')
    ax.set_xlabel('Probe at five training poses per fold; error bars show seed range.',fontsize=9)
    ax.set_ylim(-3,1.8)
    ax=axs[1,1]
    arms=['original','full','no_sentence','pose_only']
    for seed,offset in [(42,-.12),(7,0),(19,.12)]:
        values=[original.loc[original.model.eq(f'MLP seed {seed}'),'equal_recording_joint_mse'].iloc[0]]
        values += [new.loc[new.arm.eq(arm)&new.seed.eq(seed)&new.partition.eq('validation'),'joint_mse'].mean() for arm in arms[1:]]
        ax.scatter(np.arange(4)+offset,values,color=COLORS[seed],s=42,label=f'Seed {seed}')
        if seed==42:
            for i,v in enumerate(values): ax.annotate(f'{v:.2f}',(i+offset,v),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    fixed=original.loc[original.model.eq('Ridge'),'equal_recording_joint_mse'].iloc[0]
    tuned=new.loc[new.arm.eq('ridge')&new.partition.eq('validation'),'joint_mse'].mean()
    ax.axhline(fixed,color='#4c565d',lw=1.5,label=f'Fixed ridge: {fixed:.2f}')
    ax.axhline(tuned,color='#4c565d',lw=1,ls='--',label=f'Grouped ridge: {tuned:.2f}')
    ax.set_yscale('log'); ax.set_ylim(1.9,65)
    ax.set_xticks(range(4),['Fixed MLP\nfull inputs','Grouped α\nfull inputs','Grouped α\npose + signer','Grouped α\npose only'])
    ax.set_ylabel('Equal-recording joint MSE'); ax.set_title('D  Joint MSE after grouped selection',loc='left')
    handles,labels=ax.get_legend_handles_labels()
    ax.legend(handles[-2:],labels[-2:],frameon=False,fontsize=9,loc='upper right')
    fig.suptitle('MLP errors on held-out OpenFace recordings',fontsize=17,x=.05,ha='left',weight='bold')
    fig.text(.05,.93,'4,414 neutral frames · 63 recordings · 8 known signers · 10 known sentences · 0 held-out signer–sentence pairs seen in training',fontsize=11)
    from matplotlib.lines import Line2D
    common=[Line2D([0],[0],marker='o',linestyle='',color=COLORS[s],label=f'Seed {s}') for s in [42,7,19]]
    common.extend([Line2D([0],[0],marker='o',linestyle='',color='#5b6268',label='Inner target (A)'),Line2D([0],[0],marker='x',linestyle='',color='#5b6268',label='Outer target (A)')])
    fig.legend(handles=common,loc='upper left',bbox_to_anchor=(.05,.915),ncol=5,frameon=False,fontsize=9)
    fig.text(.05,.016,'A, B and D use identical held-out recordings. Follow-ups are exploratory; settings are selected inside outer training folds. C checks model responses, not causal effects.\nSource: Kuznetsova et al. (2022), pinned released OpenFace inputs; local fixed experiment and separate analysis.',fontsize=9,color='#4c565d')
    fig.subplots_adjust(top=.865,bottom=.13,left=.07,right=.98,wspace=.3,hspace=.4)
    fig.savefig(figures/'diagnostic_figure.png',dpi=220)
    fig.savefig(figures/'diagnostic_figure.svg')
    plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--results", type=Path, help="Fitted-result directory; default: retained exploratory results")
    parser.add_argument("--diagnostics", type=Path, help="Diagnostic tables; default: retained diagnostic tables")
    parser.add_argument("--probe", type=Path, help="Identity-probe tables; default: retained diagnostic tables")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory; existing paths are rejected")
    args = parser.parse_args()
    results = args.results or args.repo / "results/openface/mlp-followup"
    diagnostics = args.diagnostics or args.repo / "results/openface/mlp-followup/diagnostics"
    run(args.repo, results, diagnostics, args.probe or diagnostics, args.output)
