"""Rebuild OpenFace figures; optionally rebuild the local-only pilot figure."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
NAVY='#183b50'; BLUE='#267a9b'; ORANGE='#c77732'; INK='#253441'

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv(p): return pd.read_csv(ROOT/p)
def build(pilot_recording=None):
    if pilot_recording is not None and (not pilot_recording or Path(pilot_recording).name != pilot_recording):
        raise ValueError("Pilot recording must be a single local recording ID")
    if pilot_recording is not None:
        for suffix in ["landmarks.npz", "measurements.csv"]:
            path = ROOT/f"data/cache/{pilot_recording}_{suffix}"
            if not path.is_file():
                raise FileNotFoundError(f"Optional pilot figure requires local-only cache: {path}; see docs/reproduction.md")
    historical=csv('results/openface/frame-split/historical_summary.csv')
    historical=historical[historical.partition.eq('validation')]
    whole=csv('results/openface/recording-split/summary.csv')
    recs=csv('results/openface/recording-split/recording_metrics.csv')
    recs=recs[recs.partition.eq('validation')]
    def hv(model,seed=None):
        x=historical[historical.model.eq(model)]
        if seed is not None:x=x[x.seed.eq(seed)]
        return float(x.iloc[0].joint_mse)
    def wv(model,seed=None):
        x=whole[whole.model.eq(model)]
        if seed is not None:x=x[x.seed.eq(seed)]
        return float(x.iloc[0].equal_recording_joint_mse)
    models=[('Global mean','global',None),('Signer mean','signer',None),('Full ridge','ridge',None),('Identifier ridge','identifier_ridge',None),('MLP 42','mlp',42),('MLP 7','mlp',7),('MLP 19','mlp',19)]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
    fdir=ROOT/'figures';fdir.mkdir(exist_ok=True)
    if pilot_recording is not None:
        # Figure 1 is a real selected-landmark projection, without a source photograph.
        rid=pilot_recording;cache=ROOT/f'data/cache/{rid}_landmarks.npz';a=np.load(cache)
        measure=csv(f'data/cache/{rid}_measurements.csv');index=2
        points=a['landmarks'][index]*np.array([float(a['width']),float(a['height']),float(a['width'])])
        eye=points[[133,362]];brows=points[[55,285]];v=eye[1]-eye[0]
        foot=eye[0]+np.outer(((brows-eye[0])@v)/(v@v),v)
        distances=np.linalg.norm(brows-foot,axis=1)
        np.testing.assert_allclose(distances.mean(),measure.iloc[index].inner_brow_distance_px,rtol=1e-12,atol=1e-12)
        scale=float(measure.loc[measure.reference_frame & measure.valid_geometry, 'eye_scale_px'].median())
        np.testing.assert_allclose(distances.mean()/scale,measure.iloc[index].inner_brow_height,rtol=1e-12,atol=1e-12)
        fig,(ax,trace)=plt.subplots(1,2,figsize=(10.2,3.25),gridspec_kw={'width_ratios':[1,1.25]})
        ax.plot(eye[:,0],eye[:,1],color=INK,lw=2,label='Eye-corner line')
        for i,(p,q) in enumerate(zip(brows,foot)):
            ax.plot([p[0],q[0]],[p[1],q[1]],color=ORANGE,lw=1.8,ls='--')
            ax.text(p[0],p[1]-5,['55','285'][i],ha='center',fontsize=9)
        ax.scatter(brows[:,0],brows[:,1],color=ORANGE,s=40,zorder=3,label='Inner brow landmarks')
        ax.scatter(eye[:,0],eye[:,1],color=INK,s=28)
        for p,label in zip(eye,['133','362']):ax.text(p[0],p[1]+8,label,ha='center',fontsize=9)
        ax.set_xlim(min(eye[:,0].min(),brows[:,0].min())-15,max(eye[:,0].max(),brows[:,0].max())+15);ax.set_ylim(max(eye[:,1].max(),brows[:,1].max())+20,min(eye[:,1].min(),brows[:,1].min())-20)
        ax.set_aspect('equal');ax.set_axis_off();ax.set_title('Selected frame: 0.08 s',loc='left',fontsize=11,pad=12)
        fig.text(.06,.025,f'3D mean distance / reference eye span = {distances.mean()/scale:.4f}\n2D projection; dashed segments depict 3D perpendiculars',fontsize=8)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.10),frameon=False,fontsize=8)
        trace.plot(measure.timestamp_s,measure.inner_brow_height,color=BLUE,lw=1.5,label='Raw brow proxy')
        missing=~measure.detected.astype(bool)
        for i in np.flatnonzero(missing):trace.axvspan(measure.timestamp_s.iloc[i],measure.timestamp_s.iloc[i]+.04,color='#dce2e6',lw=0)
        trace.set(xlabel='Decoded video time (s)',ylabel='Brow height / reference eye span');trace.set_title('Far neutral downward clip',loc='left',fontsize=11)
        trace.text(.98,.97,f'Detected: {measure.detected.mean():.1%}\nMissing frames stay missing',ha='right',va='top',transform=trace.transAxes,fontsize=9)
        fig.subplots_adjust(left=.06,right=.98,top=.88,bottom=.27,wspace=.42)
        fig.savefig(fdir/'01_landmark_measurement.png',dpi=190);plt.close(fig)
    # Figure 2 uses independent panels because both validation and weighting differ.
    labels=[m[0] for m in models];y=np.arange(len(models))
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.65),sharey=True)
    frame=[hv(m,s) if m!='identifier_ridge' else np.nan for _,m,s in models];recording=[wv(m,s) for _,m,s in models]
    for ax,values,title,color in zip(axes,[frame,recording],['Historical frame splits','Held-out whole recordings'],[BLUE,ORANGE]):
        ax.barh(y,values,color=color,height=.58)
        for i,value in enumerate(values):
            if np.isfinite(value):ax.text(value+max(values)*.02,i,f'{value:.3f}',va='center',fontsize=8.5)
            else:ax.text(.35,i,'Not fitted',va='center',fontsize=8.5,color='#697782')
        ax.set_title(title,loc='left',fontsize=11);ax.set_xlim(0,max(v for v in values if np.isfinite(v))*1.22);ax.set_xlabel('Joint MSE in released units squared');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis()
    fig.text(.09,.02,'Left: mean of frame-weighted fold MSEs, shared recordings.  Right: equal mean over 63 held-out recordings.',fontsize=8.3)
    fig.subplots_adjust(left=.16,right=.97,top=.88,bottom=.19,wspace=.18)
    fig.savefig(fdir/'02_validation_comparison.png',dpi=190);plt.close(fig)
    # Figure 3 retains each recording and every fixed seed; no ranking used for fitting.
    def group(model,seed=None):
        x=recs[recs.model.eq(model)]
        if seed is not None:x=x[x.seed.eq(seed)]
        return x.set_index('video_name').joint_mse
    ridge=group('ridge');signer=group('signer');mlp=group('mlp',42)
    order=(mlp-ridge).sort_values().index
    mapping=pd.DataFrame({'display_rank':np.arange(1,len(order)+1),'video_name':order})
    mapping.to_csv(fdir/'recording_order.csv',index=False)
    better=int(((mlp-ridge)<0).sum())
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.4));rank=np.arange(1,64)
    axes[0].bar(rank,(ridge-signer).loc[order],color=BLUE,width=.85)
    for seed,c,marker in [(42,ORANGE,'o'),(7,'#267a9b','x'),(19,'#7a668f','+')]:
        axes[1].scatter(rank,(group('mlp',seed)-ridge).loc[order],s=15,color=c,marker=marker,label=f'Seed {seed}')
    for ax in axes:
        ax.axhline(0,color=INK,lw=.8);ax.set_xlim(0,64);ax.set_xlabel('Recording rank (ordered by MLP 42 minus ridge)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_title('Ridge minus signer mean',loc='left',fontsize=11);axes[1].set_title('MLP minus ridge: all seeds',loc='left',fontsize=11)
    axes[0].set_ylabel('Difference in recording joint MSE');axes[1].legend(frameon=False,fontsize=8,ncol=3,loc='upper left')
    fig.text(.08,.015,'Negative values favour the first model. Exact recording identities and order: figures/recording_order.csv.',fontsize=8.3)
    fig.subplots_adjust(left=.09,right=.98,top=.87,bottom=.2,wspace=.23)
    fig.savefig(fdir/'03_recording_differences.png',dpi=190);plt.close(fig)
    inputs = [
        ROOT/'results/openface/frame-split/historical_summary.csv',
        ROOT/'results/openface/recording-split/summary.csv',
        ROOT/'results/openface/recording-split/recording_metrics.csv',
        Path(__file__),
    ]
    outputs = [fdir/'02_validation_comparison.png', fdir/'03_recording_differences.png', fdir/'recording_order.csv']
    if pilot_recording is not None:
        inputs.extend([cache, ROOT/f'data/cache/{rid}_measurements.csv'])
        outputs.insert(0, fdir/'01_landmark_measurement.png')
    metadata = {
        'input_sha256': {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        'output_sha256': {str(p.relative_to(ROOT)): digest(p) for p in outputs},
        'independent_human_validation': False,
    }
    if pilot_recording is not None:
        metadata['figure1_geometry_recalculation_passed'] = True
    (fdir/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Built OpenFace figures and recording identity map' + (' and local-only pilot figure' if pilot_recording else ''))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-recording', help='Local recording ID for the optional pilot geometry/trace figure')
    build(parser.parse_args().pilot_recording)
