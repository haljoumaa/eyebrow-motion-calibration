# Reproduction

## What is available

| Task | Required inputs |
|---|---|
| Check saved OpenFace predictions and metrics | Pinned OpenFace inputs, retrieved below |
| Rerun the fixed OpenFace experiments or exploratory follow-up | The same inputs and the recorded numerical environment |
| Run the supplied ridge inference example | Included bundle and example CSV; recorded bundle runtime |
| Check or replay the Face Landmarker pilot | Original local pilot inputs and, for checking, saved predictions and measurement caches |
| Check the published 2024 tables | Released intermediate deviations, retrieved separately |

Pilot videos, reviewed manifests, reference intervals, annotations, frame-level results, and landmark caches are excluded from this release. The aggregate pilot summary is included. All output directories in the commands below must be fresh; choose a new name when repeating a procedure.

## Installation

Run from the repository root. Python 3.11–3.12 is supported. Historical runs used CPython 3.12.5 on macOS Apple Silicon with the versions in `requirements-lock.txt`. Use that environment for numerical replay; equivalence on other platforms must be checked. Saved bundles require their recorded Python and package versions.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pip check
```

`requirements.txt` lists direct dependencies; the full lock also includes test dependencies. The exploratory follow-up can use `scripts/mlp_followup/requirements-lock.txt` in a separate CPython 3.12.5 environment, but the full lock is needed for the complete test suite and pilot extraction. Installation and retrieval need network access. OpenFace fitting and saved-weight checks use one BLAS thread.

## Retrieve OpenFace inputs and run tests

Working directory: repository root, with the environment above active.

```bash
python scripts/fetch_data.py --group openface
python -B -m pytest -q -p no:cacheprovider
```

Retrieval writes inputs under `research/upstream-review/` and checks SHA-256 hashes against `research/upstream-assets.json`, pinned to upstream commit `c44f4ba6d3b385cfb210fc6df80c8d2c00ef27f0`. An OpenFace executable is not required.

The loader selects 4,414 neutral frames and retains the 65 confidence-below-0.8 flags. The source study reports replacing low-confidence measurements with neighbouring values in [Section 2.1](https://aclanthology.org/2022.signlang-1.19.pdf), so the retained flags do not establish that the released coordinates were unchanged. Retrieve inputs before testing. Six data-dependent tests skip when inputs are absent. A complete run should pass without those skips.

## Check saved OpenFace results

Start at the repository root with retrieved inputs. This creates a verification copy and restores observations alongside the saved predictions.

```bash
python scripts/prepare_verification_checkout.py --output tmp/check
cd tmp/check
export PYTHONPATH="$PWD/src"
python scripts/verify_openface.py
python -B scripts/mlp_followup/verify_followup.py
```

The helper verifies the input hashes, restores the neutral subset, and joins observations to 123,592 fixed-experiment and 176,560 follow-up prediction rows. It checks restored CSV hashes without fitting models. The two verification commands check saved weights, splits, preprocessing, and metric arithmetic; success is reported as `PASS` or JSON with `"passed": true`.

`PYTHONPATH` selects the copy's source instead of the parent checkout's editable installation. For inputs already stored elsewhere, add `--inputs /path/to/input-root` to the helper; files must follow the paths in `research/upstream-assets.json`.

## Rerun fixed OpenFace experiments

Working directory: the verification copy created above, with its `PYTHONPATH` set.

```bash
python -m facial_motion.openface_baselines --output tmp/repeat/baselines
python -m facial_motion.openface_reproduction --output tmp/repeat/frame-split
python -m facial_motion.openface_recording --output tmp/repeat/recording-split --save-bundles
```

These produce baseline tables, frame-split tables and fit records, and whole-recording predictions, metrics, and inference bundles. Compare the new tables with the corresponding directories under `results/openface/`; timestamps and fit times will differ. Frame-split weights were not archived, so that experiment requires refitting.

Frame validation uses four `ShuffleSplit` partitions with test size 0.25 and random state 0. Whole-recording validation uses four folds in `configs/recording-folds.csv`. Fixed ridge uses alpha 0.001; the fixed MLP uses 40 ReLU units, Adam, alpha 0.0001, and at most 1,500 iterations. Seed 42 is primary, with seeds 7 and 19 for sensitivity. Settings are in `configs/`.

## Exploratory follow-up

Working directory: the verification copy, with restored fixed predictions and neutral rows. Grouped inner validation selects regularization and MLP inputs within each outer training fold. The grids and fit settings are recorded in `results/openface/mlp-followup/plan.json`.

To rebuild summaries and figures from retained fits:

```bash
RUN="$PWD/tmp/followup-summary"
python -B scripts/mlp_followup/retrieve_inputs.py --output "$RUN/inputs"
python -B scripts/mlp_followup/diagnostics.py --inputs "$RUN/inputs" --output "$RUN/diagnostics"
python -B scripts/mlp_followup/model_probe.py --output "$RUN/probe"
python -B scripts/mlp_followup/verify_followup.py --output "$RUN/verification"
python -B scripts/mlp_followup/summarize.py --diagnostics "$RUN/diagnostics" --probe "$RUN/probe" --output "$RUN/summary"
```

To refit all 184 models and generate new summaries:

```bash
RUN="$PWD/tmp/followup-repeat"
python -B scripts/mlp_followup/retrieve_inputs.py --output "$RUN/inputs"
python -B scripts/mlp_followup/diagnostics.py --inputs "$RUN/inputs" --output "$RUN/diagnostics"
python -B scripts/mlp_followup/followup.py --inputs "$RUN/inputs" --output "$RUN/results"
python -B scripts/mlp_followup/model_probe.py --results "$RUN/results" --output "$RUN/probe"
python -B scripts/mlp_followup/verify_followup.py --results "$RUN/results" --output "$RUN/verification"
python -B scripts/mlp_followup/summarize.py --results "$RUN/results" --diagnostics "$RUN/diagnostics" --probe "$RUN/probe" --output "$RUN/summary"
```

Summaries appear in `$RUN/summary/diagnostics/` and PNG/SVG figures in `$RUN/summary/figures/`. Verification should report `"passed": true`. Compare scores, weights, preprocessing, selected settings, and convergence status; timings and SVG identifiers may differ. Each step rejects existing output directories. Outputs inside the repository must be under ignored `tmp/`.

Input-arm identifiers are `full` (pose + signer + sentence), `no_sentence` (pose + signer), and `pose_only` (pose). The separate fixed `identifier_ridge` uses signer and sentence indicators without pose.

## Ridge inference example

Working directory: repository root or verification copy, using the recorded bundle runtime.

```bash
python -m facial_motion.openface_bundle \
  --bundle results/openface/recording-split/bundles/fold0_ridge.json \
  --input examples/openface-input.csv \
  --output tmp/ridge-predictions.csv
```

The example uses synthetic pose values and known categories. Output is a CSV of predicted neutral distances, preserving input order with zero-based `input_position`. Correction subtracts those predictions from observed eyebrow distances.

Required input columns are `pose_Rx`, `pose_Ry`, `pose_Rz`, `pose_Tx`, `pose_Ty`, `pose_Tz`, `speaker`, and `sentence`. Rotations are radians; translations retain released units. Numerical values must be finite and categories nonempty. Unknown signers are rejected; unknown sentences receive zero indicators before scaling and are flagged. Observed target columns are ignored.

The supplied bundle is a fold-specific evaluation fit. Use `--expected-sha256` to check the hash listed in `bundle_manifest.json`. Other fold bundles can be generated with `--save-bundles` in the recording experiment.

## Face Landmarker pilot with original local inputs

Working directory: repository root. Supply the original reviewed `data/manifest.csv`, `data/references.csv`, and `data/annotations.csv`. The manifest supplies video URLs and hashes. Retrieval includes the model pinned in `models/model_metadata.json`.

```bash
python scripts/fetch_data.py --group pilot
python scripts/extract_pilot.py
python -m facial_motion evaluate --output tmp/repeat/face-landmarker
```

Extraction writes landmarks and measurements to `data/cache/` and extraction metadata to `results/pilot/`. Evaluation writes predictions, recording-level metrics, a summary, and run metadata to the requested directory. It holds out each camera distance in turn and checks source and model hashes.

`results/face-landmarker/summary.csv` contains mean recording-level RMSE, MAE, and AUROC by camera distance, condition, method, and evaluation scope. `supported_pose` scores frames within the training pitch range; `complete_clip_raw_fallback` retains raw measurements outside it. Alternating clips use AUROC to distinguish annotated raised and neutral states.

The [pooled pilot table](../results/face-landmarker/pooled_summary.csv) gives equal-recording averages across camera distances at full numerical precision for both scopes and all five methods. Its recording counts were checked against the original saved metrics. The [provenance record](../research/pilot-summary-provenance.json) records the source hashes and counts by distance, allowing the pooled averages to be reconstructed from the distance-specific summary.

To check saved pilot arithmetic, supply the original pilot input/result files and measurement caches. Start at the repository root:

```bash
python scripts/prepare_verification_checkout.py \
  --inputs /path/to/private-input-root \
  --pilot-cache /path/to/data/cache \
  --output tmp/pilot-check
cd tmp/pilot-check
export PYTHONPATH="$PWD/src"
python scripts/verify_pilot.py
```

The private input root must also contain the pinned OpenFace inputs. Pilot verification reports `PASS` after checking saved predictions against caches and recalculating metrics. The aggregate summary alone cannot support this check.

For a combined replay from the measurement caches, the copy also needs the external model and original extraction metadata under their recorded paths. Then `python scripts/reproduce.py --output tmp/fixed-repeat` reruns all fixed OpenFace and pilot experiments, compares tables and fitted state at `1e-12` relative/absolute tolerance, and reloads bundles in a separate process. Run the exploratory follow-up separately.

## Figures and published-table checks

Working directory: verification copy.

```bash
python scripts/build_figures.py
python scripts/fetch_data.py --group literature
python scripts/reproduce_published_tables.py
```

Figure generation writes the two fixed OpenFace figures and provenance to `figures/`. The literature command checks 48 published table entries from released intermediate deviations and reports `PASS`; it does not rerun video extraction or significance tests.

The follow-up summary commands above rebuild the diagnostic figure. For local pilot figures, `build_figures.py --pilot-recording LOCAL_RECORDING_ID` requires matching landmark NPZ and measurement CSV files; the verification helper copies measurement caches only. `python -m facial_motion experiment-figures --recording-ids ID1 ID2 ID3` uses three locally selected pilot recordings.

## Data use

The [MIT License](../LICENSE) covers original code in `src/`, `scripts/`, and `tests/`. Datasets, research outputs, figures, fitted model artifacts, dependencies, and third-party models have separate terms.

OpenFace inputs are retrieved from the [pinned upstream repository](https://github.com/kuzanna2016/non-manuals-2021/tree/c44f4ba6d3b385cfb210fc6df80c8d2c00ef27f0); its inspected READMEs contain no dataset license. Pilot videos are sourced from [OSF](https://osf.io/jvc42/); the inspected project has no selected project license. Consult the source terms before reuse. Recording codes alone do not establish anonymity.

The [bibliography](../research/references.bib) credits the source studies, OpenFace 2.0, and the facial geometry work.
