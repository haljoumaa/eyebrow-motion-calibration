# Methodology

The OpenFace experiments evaluate pose-dependent neutral eyebrow predictions. The separate Face Landmarker pilot evaluates pitch correction and expression preservation using OSF videos. Different geometry and measurement scales prevent comparison of their scores.

## OpenFace neutral-distance experiments

The experiments start from measurements released by Kuznetsova et al. (2022), pinned to commit `c44f4ba6d3b385cfb210fc6df80c8d2c00ef27f0`. They evaluate neutral-distance prediction; new landmark extraction and functional PCA are outside their scope.

### Inputs and models

The loader selects neutral rows from `open_face_with_distances.csv` using `additional_stats_from_elan.json`, preserving source order: 4,414 frames, 63 recordings and eight signers. It retains all 65 frames with confidence below 0.8 and all tracking-success flags, without quality exclusions.

Targets are `inner_perp_dist39_42_3d` and `outer_perp_dist39_42_3d` in released coordinate units, without conversion to millimetres. Numerical inputs are rotation, cosine of rotation and translation for x, then y, then z; rotations are radians. Signer and sentence indicators follow, giving 27 columns on these folds. Categories and standardization of the complete matrix are fitted on training rows only. Targets are unscaled.

Predictions estimate neutral distances. A corrected observation is `observed_distance - predicted_neutral_distance`.

Controls are global and signer means weighted by training frames. Full-input ridge uses alpha 0.001. The recording-split experiment also includes ridge with signer and sentence indicators but no pose features (`identifier_ridge`), using the same alpha. The two-output MLP uses 40 ReLU units, Adam, alpha 0.0001, at most 1,500 iterations and no early stopping. Seed 42 is primary; seeds 7 and 19 assess optimization sensitivity. A convergence warning marks an incomplete fit; the fixed protocol permits one supplementary fresh fit with the same seed and 3,000 iterations. All fitting and scoring use one BLAS thread.

### Validation

Four `ShuffleSplit` frame partitions use test size 0.25 and random state 0, with 3,310 training and 1,104 validation rows each. Every partition shares all 63 recordings across training and validation. Validation rows can recur across splits. Summaries average the four frame-weighted scores.

`configs/recording-folds.csv` fixes four recording folds with 17, 17, 15 and 14 held-out recordings; its checksum is checked before fitting. Each recording is held out once, with no recording overlap within a fold. Every validation signer is represented in training. Models and preprocessors are fitted separately with equal frame weights. Primary scores weight the 63 held-out recordings equally; frame-weighted and equal-signer summaries are also retained. The differing summary weights prevent a direct percentage comparison with frame-split scores.

Reusable bundles reject unknown signers. Unknown sentences receive zero indicators before scaling and are flagged. Validation has no unseen sentences, so this handling rule is not evidence of generalization. Bundles are evaluation-fold fits. Transfer, anatomical accuracy and expression preservation are untested.

### Retained evidence

Run records contain source and input hashes, row identities, fitted categories and scalers, settings, warnings and convergence status. Recording-split records also retain fitted weights; frame-split estimator weights were not archived. Source-study reference scores are in `results/openface/frame-split/archived_reference.json` and `reproduction_comparison.json`, separate from the local experiment tables. Saved-weight checks reconstruct predictions and recompute metrics from residuals without fitting.

## Exploratory MLP follow-up

The exploratory follow-up examines the fixed MLP's held-out errors using the same 4,414 neutral frames, 63 recordings, eight signers, ten sentences and four outer folds. Joint MSE averages the two target MSEs within each recording; primary summaries weight recordings equally. Training summaries weight recording–fit pairs equally; each recording contributes three training fits.

### Grouped selection

The MLP retains 40 ReLU units, Adam, 1,500 iterations, no early stopping and unscaled targets. Its input arms are MLP (pose + signer + sentence), MLP (pose + signer), and MLP (pose only). All retain the nine numerical pose features defined above; the second arm excludes sentence identity. Each arm considers alpha {0.0001, 1, 100}; full-input ridge considers {0.001, 1, 100}. Three recording-grouped inner folds within each outer training set select alpha by pooled equal-recording joint MSE at seed 42. Ties prefer stronger regularization. The same inner scores select the MLP input arm. Encoders and scalers are fitted afresh on each training partition; seeds 7 and 19 do not influence selection.

Historical tables and the diagnostic figure label the `no_sentence` arm “Signer only”; this denotes pose + signer, not signer indicators alone. “Full indicators” denotes pose + signer + sentence. These names differ from identifier-only ridge, which excludes pose features.

The workflow contains 108 inner and 36 outer MLP fits, plus 40 ridge fits. One inner pose-only fit reached the iteration limit, making that alpha candidate ineligible for selection in its outer fold; all outer fits completed. Every MLP arm selected alpha 100, the edge of the bounded grid. The selected policy chose MLP (pose + signer) in every fold. This is exploratory because the original held-out findings informed the follow-up design.

### Diagnostics

Recording-level residual means distinguish offset errors from within-recording fluctuations. Pose coverage checks all six rotations/translations against training-only ranges. Nearest-neighbor novelty uses training-only standard or robust IQR scaling across these six features, normalizing Euclidean distance by √6. A descriptive threshold is the frame-weighted 95th percentile of training-neighbor distances from other recordings. Global and same-signer coverage are retained. Cosines are omitted from these distances because they are deterministic rotation transforms.

Associations use Spearman correlations and rank residualization against fold indicators, frame-count rank and mean-confidence rank. Quality sensitivity scores confidence ≥0.8 frames without changing training. Sixty released frames have absolute translations above one million, inflating standard deviations relative to IQRs; their influence on fitting was not isolated.

An identity-response probe crosses known signer/sentence categories at five actual central training poses per fold, removes additive category means and summarizes interaction RMS with equal pose/fold weights. All held-out signer–sentence pairs are unseen while individual categories are known; each pair corresponds to one recording. Pair novelty and recording holdout therefore cannot be separated. Synthetic category swaps are not causal interventions or anatomical measurements.

### Interpretation and limitations

At seed 42, the selected MLP has lower joint MSE than fixed ridge on 27 of 63 recordings, with median paired difference (MLP minus ridge) +0.177. Recording offsets and the probe are consistent with overfitting to signer–sentence combinations, but do not establish the cause of the errors. Translation extremes remain a competing explanation.

The bounded grid does not establish either model family's best performance. Correlated frames, eight signers, approximate neutral targets and inseparable pair/recording identity limit causal interpretation. Confirmatory evaluation requires fresh recordings. The [comparison table](../results/openface/mlp-followup/diagnostics/main_table.csv) and [historical diagnostic figure](../figures/mlp-followup/diagnostic_figure.png) retain the original labels; the figure's explanatory titles should be read as descriptive interpretations. Commands are in the [reproduction guide](reproduction.md).

## Face Landmarker pilot

The pilot evaluates pitch correction on 22 recordings of one participant: 11 neutral, six held raised and five alternating. Twelve yaw/roll videos are outside scope. Training and evaluation use separate camera distances. Both pose and eyebrow position are estimated by the same vision model.

### Measurement and labels

Face Landmarker Tasks runs in video mode with one face and a fresh tracker per recording. `models/model_metadata.json` pins the model URL and hash. Coordinates are proportional to `(x * width, y * height, z * width)`, an approximate relative reconstruction.

The primary measure averages the perpendicular 3D distances from brow landmarks 55/285 to the 133–362 eye-corner line. The fixed scale is the median 33–263 eye span in the recording's reference segment. The cached outer-brow measure at 46/276 is outside the primary comparison.

Rotation comes from polar decomposition of the transformation matrix's 3×3 block; singular and reflected matrices are rejected. Extrinsic `xyz` Euler decomposition uses `R = Rz(roll) Ry(yaw) Rx(pitch)`. Relative pose uses `R_reference.T @ R_frame`, with the mean rotation of valid reference frames. Positive pitch denotes downward movement, checked against visible movement. Missing detections and invalid geometry remain missing, without interpolation.

Reference selection uses a stable frontal segment chosen before inspecting correction outputs. The preferred duration is 0.5 s, shortened when necessary to at least 0.2 s and five valid frames. Eyebrows may be neutral or raised. References last 0.20–0.50 s; `data/references.csv` records reasons. One middle-distance upward alternating clip was excluded before inspecting correction results because it lacked a usable reference. No clip was removed for low detection coverage.

`data/annotations.csv` records conservative visible-state intervals, excluding transitions and uncertain episodes. Half-open intervals use a 0.2 s transition margin. The five included alternating clips contain nine clear raise episodes; possible ambiguous second raises remain excluded. The author review is retained in `data/audit_review.json`; no independent second-rater validation or agreement estimate exists. `data/qa_frames.csv` records selected review times.

### Models and validation

Neutral recordings use target `h(t) - median(h_reference)`, predicted from pitch. Input and target scaling use training data only. Comparisons include raw measurements, a Gaussian GAM with a cubic penalized spline (six basis functions; penalties 0.1/1/10), and an MLP (eight tanh units, L-BFGS, penalties 0.001/0.01/0.1, no automatic early stopping). Seed 42 is primary; seeds 7 and 19 assess sensitivity. A convergence failure invalidates a candidate; warnings are retained.

Each training recording contributes the same number of evenly spaced valid frames: `min(150, smallest valid recording count in that partition)`, preserving temporal order. Correction is `h_raw - [g(p) - g(0)]`, retaining the raised recording's reference level.

Each camera distance (close, middle or far) is held out once, with all its recordings. Fitting and tuning use neutral recordings at the other two distances. Inner validation trains at one distance and validates at the other, then reverses. Lowest mean recording-level supported-pose RMSE selects the candidate; grid order breaks ties. The selected model is refitted on both training distances.

Whole recordings, matching file hashes and overlap groups cannot cross partitions. The 22 included clips retain separate content groups after screening 15 within-distance neutral pairs for sustained overlap; transformed overlap could still be missed.

### Metrics and interpretation

Supported pitch is the minimum-to-maximum range of balanced training observations. Primary comparisons use identical valid, supported, non-reference masks. Complete-clip metrics use raw fallback outside support. Detection coverage and support fractions are retained.

Held-neutral and held-raised recordings have separate reference-relative MAE/RMSE. Alternating clips use visible-state AUROC with fixed score orientation and matched availability masks; flat predictions score 0.5 when both states occur. Summaries weight recordings equally and retain individual metrics and paired raw differences. No frame-level significance tests or population p-values are used.

The study includes reference calibration at evaluation time. Startup jitter, imperfect frontal references, missing detections, one participant and model-based geometry limit interpretation. Frontal pose/scale were not independently measured; repeatable inference does not establish anatomical accuracy or expression preservation.

## Sources and attribution

The correction method and released OpenFace inputs follow [Kuznetsova et al. (2022)](https://aclanthology.org/2022.signlang-1.19/) and the [pinned input repository](https://github.com/kuzanna2016/non-manuals-2021/tree/c44f4ba6d3b385cfb210fc6df80c8d2c00ef27f0). The pilot is motivated by [Kuznetsova and Kimmelman (2024)](https://arxiv.org/abs/2403.10367v2) and the [OSF study](https://osf.io/jvc42/), but uses Face Landmarker Tasks rather than Holistic. The [bibliography](../research/references.bib) also credits OpenFace 2.0 and the facial geometry work.
