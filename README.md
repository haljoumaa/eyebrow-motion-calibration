# Eyebrow measurement under head movement

This project investigates whether head-pose correction improves eyebrow measurements used in sign-language analysis without suppressing expression. It predicts neutral eyebrow distances from released OpenFace measurements, compares frame-level and whole-recording validation, and explores MLP regularization and input ablations. A separate Face Landmarker pilot evaluates pitch correction.

## Principal findings

The fixed OpenFace experiments use 4,414 neutral frames from 63 recordings and eight signers. Ridge regression and a multilayer perceptron (MLP) use pose, signer and sentence inputs. Joint mean squared error (MSE) averages the inner- and outer-brow target errors; seed 42 is the primary MLP fit.

| Fixed evaluation | Ridge joint MSE | MLP joint MSE |
|---|---:|---:|
| Frame splits, mean frame-weighted fold score | 1.317580 | 0.429238 |
| Whole-recording folds, equal recording weights | 2.471308 | 31.796760 |

Frame splits share recordings between training and validation; the different score weights prevent a direct percentage comparison. In the exploratory follow-up, recording-grouped selection gives joint MSEs of 3.106 for the full-input MLP, 2.623 for MLP (pose + signer), and 6.093 for MLP (pose only), versus 2.600 for ridge with grouped selection. The fixed ridge score remains lower at 2.471. These results do not establish an overall MLP advantage. See the [comparison table](results/openface/mlp-followup/diagnostics/main_table.csv).

In the separate one-participant, 22-recording pilot, the generalized additive model (GAM) reduces neutral root mean squared error (RMSE) by 26.9% within the training pitch range. Held-raised RMSE increases by 29.8%, and raised/neutral separation decreases. Neither the GAM nor the pilot MLP demonstrates expression preservation. See the [aggregate pilot results](results/face-landmarker/summary.csv).

OpenFace validation includes no unseen signers or sentences. Correlated frames, approximate neutral targets, the exploratory follow-up design and imperfect pilot references limit interpretation. The [methodology](docs/methodology.md) describes the datasets, models, validation and limitations.

## Figures

![Fixed frame and whole-recording validation](figures/02_validation_comparison.png)

[Recording differences](figures/03_recording_differences.png) · [MLP diagnostics](figures/mlp-followup/diagnostic_figure.png)

In the historical diagnostic figure and result tables, “Signer only” means pose + signer, with sentence identity excluded (`no_sentence`). The diagnostics describe model behavior; they do not establish the cause of the held-out errors.

## Reproduction

The [reproduction guide](docs/reproduction.md) covers installation, input retrieval, tests, saved-result checks, experiment runs, inference and figure generation. The recorded environment is CPython 3.12.5 on macOS Apple Silicon.

Code is in `src/` and `scripts/`; experiment settings are in `configs/` and `config.yaml`. Numerical outputs, fitted state and predictions are in `results/`, and figures are in `figures/`. `tests/` contains the test suite; `research/` contains pinned input manifests and [source references](research/references.bib).

Source datasets, videos, full landmark caches and external model binaries are retrieved separately. Participant-linked pilot manifests, annotations, review records, frame predictions, recording metrics, run metadata and the real-frame figure are excluded; aggregate pilot results remain. Exact pilot replay requires the original local inputs. See the [reproduction guide](docs/reproduction.md).

## License and data sources

The original code in `src/`, `scripts/` and `tests/` is licensed under the [MIT License](LICENSE), copyright © 2026 Hareth Al-Jomaa. This grant excludes datasets, research outputs, figures, fitted model artifacts and third-party materials.

The OpenFace inputs come from [Kuznetsova et al. (2022)](https://github.com/kuzanna2016/non-manuals-2021); the pilot videos come from [Kuznetsova and Kimmelman (2024)](https://osf.io/jvc42/). The [data-use notes](docs/reproduction.md#licensing-and-data-use) describe source terms, the pilot exclusions and local reproduction.
