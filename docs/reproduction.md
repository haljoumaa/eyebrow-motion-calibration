# Reproduction

## Installation and dependencies

Run from the repository root. The package declares Python 3.11–3.12 support; exact reproduction and saved bundle loading use CPython 3.12.5 and the recorded package versions on macOS Apple Silicon. `requirements.txt` lists direct dependencies; `requirements-lock.txt` pins runtime and test dependencies. Fitting and saved-weight checks use one BLAS thread. Numerical equivalence on other environments must be checked.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pip check
```

For follow-up commands alone, install `scripts/mlp_followup/requirements-lock.txt` in a separate Python 3.12.5 environment. Use the full lock for the complete test suite and pilot extraction. No OpenFace executable or upstream pickle is needed. Installation and input retrieval require network access.

## Inputs and tests

```bash
python scripts/fetch_data.py --group openface
python -B -m pytest -q -p no:cacheprovider
```

Retrieval uses the pinned commit and URLs in `research/upstream-assets.json` and checks SHA-256 hashes. Six tests require these external inputs and explicitly skip when they are absent: the four `packet` tests in `tests/test_openface_baselines.py`, `test_real_historical_folds_scaler_and_legacy_values`, and `test_fixed_config_and_real_fold_inventory`. Run the full suite after retrieval; skips do not count as scientific verification. The loader preserves source row order and all 65 low-confidence flags.

The original local pilot manifest lists 35 videos, including 23 pitch clips and 12 yaw/roll clips outside scope; 22 pitch clips were included. The publication excludes `data/manifest.csv`, `data/references.csv`, `data/annotations.csv`, `data/qa_frames.csv`, `data/audit_review.json`, `results/face-landmarker/predictions.csv`, `results/face-landmarker/recording_metrics.csv`, `results/face-landmarker/run_metadata.json` and `figures/01_landmark_measurement.png`. These link individual measurements or review decisions to identifiable source videos. The unchanged condition-level `results/face-landmarker/summary.csv` retains the aggregate experiment results; no synthetic observations replace the excluded data.

## Check saved results without fitting

The two `predictions-public.csv` tables contain 123,592 fixed-experiment and 176,560 exploratory prediction rows. Source observations and fixed-experiment residuals are omitted. Restore those columns from pinned inputs in an isolated verification copy:

```bash
python scripts/prepare_verification_checkout.py --output tmp/check
cd tmp/check
export PYTHONPATH="$PWD/src"
python scripts/verify_openface.py
python -B scripts/mlp_followup/verify_followup.py
```

Run subsequent commands from this copy. `PYTHONPATH` selects its source instead of the parent checkout's editable installation. The helper copies the repository and verified inputs, restores the 4,414-row neutral subset, and joins observations to saved predictions. It checks the restored CSVs against their original hashes without fitting or recomputing predictions.

Use `--inputs /path/to/input-root` for existing inputs with the manifest's layout. For exact pilot arithmetic, `--inputs /path/to/private-input-root --pilot-cache /path/to/data/cache` also requires the excluded original pilot inputs and result files under that input root. The helper copies them and checks input/cache hashes against the original local run metadata. Then run `python scripts/verify_pilot.py` inside the copy. Without these files, pilot arithmetic cannot be checked from the aggregate summary alone. Frame-split tables and source-study reference scores remain separate.

## Rerun fixed experiments

Choose fresh output directories:

```bash
python -m facial_motion.openface_baselines --output tmp/repeat/baselines
python -m facial_motion.openface_reproduction --output tmp/repeat/frame-split
python -m facial_motion.openface_recording --output tmp/repeat/recording-split --save-bundles
```

For the separate pilot, first supply the original reviewed manifest, reference intervals and annotations locally. Obtain the videos from the [original OSF project](https://osf.io/jvc42/); `fetch_data.py` uses the supplied manifest and checks its hashes. The following commands regenerate excluded outputs locally:

```bash
python scripts/fetch_data.py --group pilot
python scripts/extract_pilot.py
python -m facial_motion evaluate --output tmp/repeat/face-landmarker
```

Without the original reviewed inputs, `python -m facial_motion inventory` creates a provisional manifest from OSF. Set inclusion decisions, reference intervals and visible-state annotations locally using the [pilot protocol](methodology.md#face-landmarker-pilot). Source videos alone do not recover the original manual decisions, so this route cannot promise exact historical pilot values. Pilot commands fail clearly if required local artifacts are absent.

Extraction processes included clips at decoded native timestamps with a fresh tracker per clip. It writes measurements and landmarks to `data/cache/` and metadata to `results/pilot/`. The evaluator checks source and model hashes. Frame-split weights and pilot fitted estimators were not archived. OpenFace settings, scalers and numerical outputs remain; pilot settings and the aggregate summary remain, while its recorded preprocessing is in the excluded local run metadata. Model replay requires refitting. `scripts/verify_pilot.py` checks saved predictions against measurement caches matching the recorded hashes.

After restoring the complete OpenFace CSVs, the excluded original pilot inputs and result files, and matching pilot measurement caches, `python scripts/reproduce.py --output tmp/fixed-repeat` reruns and compares all fixed experiments. It uses `1e-12` relative and absolute tolerances and replays bundles in a separate process. The exploratory follow-up is run separately.

## Exploratory follow-up

Run from the verification copy above, using a fresh `RUN` directory. Diagnostics and model probing require the restored fixed predictions and neutral rows.

```bash
RUN="$PWD/tmp/followup-repeat"
python -B scripts/mlp_followup/retrieve_inputs.py --output "$RUN/inputs"
python -B scripts/mlp_followup/diagnostics.py --inputs "$RUN/inputs" --output "$RUN/diagnostics"
python -B scripts/mlp_followup/followup.py --inputs "$RUN/inputs" --output "$RUN/results"
python -B scripts/mlp_followup/model_probe.py --results "$RUN/results" --output "$RUN/probe"
python -B scripts/mlp_followup/verify_followup.py --results "$RUN/results" --output "$RUN/verification"
python -B scripts/mlp_followup/summarize.py --results "$RUN/results" --diagnostics "$RUN/diagnostics" --probe "$RUN/probe" --output "$RUN/summary"
```

Each step rejects an existing output directory. Outputs inside the repository must be under ignored `tmp/`. The [methods](methodology.md#exploratory-mlp-follow-up) describe grouped selection, 184 fits and sensitivity seeds. Summaries contain tables and PNG/SVG figures, using “MLP (pose + signer)” for the historical `no_sentence` arm. To rebuild diagnostics without fitting, omit the `followup.py` step and omit `--results` from model probing, verification and summarization; those commands then use retained fits. Supply the fresh diagnostic and probe directories as above. Compare tables, weights, preprocessing, selections and convergence; timestamps, timings and SVG identifiers can differ.

## Inference, figures and literature arithmetic

```bash
python -m facial_motion.openface_bundle \
  --bundle results/openface/recording-split/bundles/fold0_ridge.json \
  --input examples/openface-input.csv \
  --output tmp/ridge-predictions.csv
```

`examples/openface-input.csv` uses synthetic pose values for API illustration and category strings from the fitted vocabulary; it contains no experimental observations. The example requires `pose_Rx`, `pose_Ry`, `pose_Rz`, `pose_Tx`, `pose_Ty`, `pose_Tz`, `speaker` and `sentence`. Rotations are radians; translations retain released units. Subtract predicted neutral distances from observations for correction. Numeric inputs must be finite and category strings nonempty. Missing required columns, duplicate columns and invalid rows are rejected. Unknown signers are rejected; unknown sentences receive zero indicators before scaling and are flagged. Row order is preserved, with zero-based `input_position` in the output. Observed target columns are ignored. Bundles require OpenFace inputs and the recorded runtime versions.

The supplied bundle is an evaluation-fold fit; no full-data deployment fit is provided. `--expected-sha256` checks a bundle against its manifest hash. Regenerate other recording-fold bundles with `--save-bundles`; their weights and mean estimators are in `model_runs.json` and `training_means.json`. Follow-up outer-fit weights are in `fit_runs.json`, with scalers and categories in `preprocessing.json`. These records support inference, not resumed optimization.

`python scripts/build_figures.py` rebuilds the two retained OpenFace figures from saved tables. To regenerate the excluded pilot figure locally, add `--pilot-recording LOCAL_RECORDING_ID` with the original selected recording ID and matching landmark NPZ and measurement CSV. The helper does not copy landmark NPZs; supply them locally or regenerate them by extraction. The pilot figure remains outside publication; its measured geometry and time trace have not been altered. The separate `python -m facial_motion experiment-figures` command also takes `--recording-ids` with the three original locally selected trajectory examples instead of embedding video IDs in the code. Run figure generation only in an isolated copy because it writes `figures/` and provenance. The follow-up summary command rebuilds its diagnostic figure. Recording-order maps identify plotted recordings. Historical figure labels are explained in the [methods](methodology.md#grouped-selection).

Optional literature arithmetic uses `python scripts/fetch_data.py --group literature`, then `python scripts/reproduce_published_tables.py`. This checks 48 published entries from released intermediate deviations; it does not repeat raw Holistic extraction or significance tests.

## Licensing and data use

The [MIT License](../LICENSE) covers the original code in `src/`, `scripts/` and `tests/`. It does not cover datasets, research outputs, figures, fitted model artifacts or third-party materials. Dependencies and the external Face Landmarker model retain their own terms. Attribution is in [Sources and attribution](methodology.md#sources-and-attribution).

The [pinned OpenFace repository](https://github.com/kuzanna2016/non-manuals-2021/tree/c44f4ba6d3b385cfb210fc6df80c8d2c00ef27f0) has no dataset license in its published READMEs. Retrieve its checksum-pinned numerical inputs with `python scripts/fetch_data.py --group openface`. The checked recording metadata contains codes, frame rates and frame counts, without names or source-video links. No reasonable code-to-person link was established in the inspected source materials; OpenFace predictions, recording maps, diagnostics, fitted weights and preprocessing remain unchanged. This is a targeted linkage assessment, not a claim that codes guarantee anonymity. The [original collection paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0233731) also reports consent for research and publication; it does not supply a separate dataset license.

The [OSF project](https://osf.io/jvc42/) has no selected project license or documented disclosure basis for this release's participant-linked pilot artifacts. Those artifacts and their real-frame figure are excluded. The aggregate pilot summary and the 48-entry arithmetic comparison with published literature tables remain unchanged; neither contains a pilot recording-ID map or frame trace. Source references and download specifications are retained for local reproduction, subject to source terms.
