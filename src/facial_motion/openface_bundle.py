"""Versioned, data-only inference bundles for our fitted OpenFace models.

JSON stores preprocessing and fitted estimator state together. Reload never
executes a pickle or refits a scaler, category encoder, or model.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .openface_data import POSE_COLUMNS, TARGETS, sha256
from .openface_reproduction import CATEGORIES, NUMERIC_FEATURES, now, write_json


SCHEMA_VERSION = 1
MODELS = {"global", "signer", "ridge", "identifier_ridge", "mlp"}


def runtime_versions():
    return {"python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")}}


def _array(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Invalid bundle {label}: expected finite shape {shape}")
    return result


class OpenFaceBundle:
    """Inference from explicitly validated scaler, categories and learned weights."""

    def __init__(self, payload, require_runtime=True):
        try:
            self._validate(payload, require_runtime)
        except (KeyError, TypeError, IndexError, OverflowError) as error:
            raise ValueError(f"Invalid OpenFace bundle: {error}") from error
        self.payload = payload

    def _validate(self, payload, require_runtime):
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported OpenFace bundle schema version")
        self.model = payload["model"]
        if self.model not in MODELS:
            raise ValueError("Unsupported OpenFace bundle model")
        if payload["target_order"] != list(TARGETS) or payload["target_scaling"] is not False:
            raise ValueError("Invalid bundle target order or scaling")
        if payload["unknown_signer_policy"] != "reject":
            raise ValueError("Bundle must reject unknown signers")
        if payload["unknown_sentence_policy"] != "zero indicators before scaling; flag":
            raise ValueError("Invalid bundle unknown-sentence policy")
        context = payload["context"]
        for key in ("fit_id", "fold", "config", "input_provenance", "source_sha256"):
            if key not in context:
                raise ValueError(f"Missing bundle context: {key}")
        if require_runtime and payload["runtime"] != runtime_versions():
            raise ValueError("Bundle runtime differs; use the recorded pinned Python/package versions")
        prep = payload["preprocessing"]
        self.categories = prep["categories"]
        for name in CATEGORIES:
            values = self.categories[name]
            if (not isinstance(values, list) or not values or
                    any(not isinstance(v, str) or not v.strip() for v in values) or
                    values != sorted(set(values))):
                raise ValueError(f"Invalid bundle categories: {name}")
        numeric = [] if self.model == "identifier_ridge" else list(NUMERIC_FEATURES)
        expected = [*numeric, *(f"{c}_{v}" for c in CATEGORIES for v in self.categories[c])]
        if prep["numeric_feature_order"] != numeric or prep["feature_order"] != expected:
            raise ValueError("Invalid bundle feature order")
        if prep["target_order"] != list(TARGETS) or prep["target_scaling"] is not False:
            raise ValueError("Invalid preprocessing target order/scaling")
        width = len(expected)
        self.mean = _array(prep["scaler_mean"], (width,), "scaler mean")
        self.scale = _array(prep["scaler_scale"], (width,), "scaler scale")
        variance = _array(prep["scaler_var"], (width,), "scaler variance")
        if (self.scale <= 0).any() or (variance < 0).any():
            raise ValueError("Invalid bundle scaler: scale must be positive and variance nonnegative")
        if type(prep["scaler_n_samples_seen"]) is not int or prep["scaler_n_samples_seen"] <= 0:
            raise ValueError("Invalid bundle scaler training sample count")
        required = list(CATEGORIES)
        if self.model in {"ridge", "mlp"}:
            required = [*POSE_COLUMNS, *required]
        if payload["required_input_columns"] != required:
            raise ValueError("Invalid bundle required input columns")
        self.required = required
        state = payload["estimator"]
        if self.model in {"global", "signer"}:
            if state["class"] != "MeanBaseline" or state["parameters"]["mode"] != self.model:
                raise ValueError("Invalid bundle mean estimator")
            params = state["parameters"]
            if params["target_order"] != list(TARGETS):
                raise ValueError("Invalid bundle mean target order")
            self.global_mean = _array(params["global_mean"], (2,), "global mean")
            if set(params["signer_means"]) != set(self.categories["speaker"]):
                raise ValueError("Invalid bundle signer means")
            self.signer_means = {k: _array(v, (2,), "signer mean") for k, v in params["signer_means"].items()}
            counts = params["signer_counts"]
            if (set(counts) != set(self.signer_means) or
                    any(type(v) is not int or v <= 0 for v in counts.values()) or
                    sum(counts.values()) != params["n_training_frames"] or
                    params["n_training_frames"] != prep["scaler_n_samples_seen"]):
                raise ValueError("Invalid bundle mean training counts")
        elif self.model in {"ridge", "identifier_ridge"}:
            if state["class"] != "Ridge" or state["parameters"]["alpha"] != .001:
                raise ValueError("Invalid bundle ridge estimator settings")
            self.coef = _array(state["coef"], (2, width), "ridge coefficients")
            self.intercept = _array(state["intercept"], (2,), "ridge intercept")
        else:
            params = state["parameters"]
            fixed = {"hidden_layer_sizes": [40], "activation": "relu", "solver": "adam",
                     "alpha": .0001, "early_stopping": False}
            if (state["class"] != "MLPRegressor" or
                    any(params[k] != v for k, v in fixed.items()) or
                    params["random_state"] not in (42, 7, 19) or
                    params["max_iter"] not in (1500, 3000) or
                    state["out_activation"] != "identity"):
                raise ValueError("Invalid bundle MLP estimator settings")
            if len(state["coefs"]) != 2 or len(state["intercepts"]) != 2:
                raise ValueError("Invalid bundle MLP layer count")
            self.coefs = [_array(v, s, "MLP weights") for v, s in zip(state["coefs"], ((width, 40), (40, 2)))]
            self.intercepts = [_array(v, s, "MLP biases") for v, s in zip(state["intercepts"], ((40,), (2,)))]

    def _inputs(self, rows):
        if not isinstance(rows, pd.DataFrame) or rows.empty:
            raise ValueError("Prediction inputs must be a nonempty pandas DataFrame; no rows are dropped")
        if rows.columns.duplicated().any():
            raise ValueError("Duplicate prediction input columns are not supported")
        missing = set(self.required) - set(rows.columns)
        if missing:
            raise ValueError(f"Missing OpenFace input columns: {sorted(missing)}")
        values = rows.loc[:, list(CATEGORIES)]
        if not values.map(lambda x: isinstance(x, str) and bool(x.strip())).all().all():
            raise ValueError("Signer/sentence identifiers must be nonempty strings")
        unknown = ~rows.speaker.isin(self.categories["speaker"])
        if unknown.any():
            raise ValueError(f"Unknown signer: personalized OpenFace models require a known signer; {sorted(rows.loc[unknown, 'speaker'].unique())}")
        return ~rows.sentence.isin(self.categories["sentence"]).to_numpy()

    def design(self, rows):
        self._inputs(rows)
        blocks = []
        if self.model in {"ridge", "mlp"}:
            try:
                pose = rows.loc[:, list(POSE_COLUMNS)].to_numpy(dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError("OpenFace pose inputs must be numeric and finite") from error
            if not np.isfinite(pose).all():
                raise ValueError("OpenFace pose inputs must be numeric and finite; no rows are dropped")
            blocks.extend(value for i in range(3) for value in (pose[:, i], np.cos(pose[:, i]), pose[:, i + 3]))
        for category in CATEGORIES:
            blocks.extend(rows[category].eq(value).to_numpy(dtype=float) for value in self.categories[category])
        return np.column_stack(blocks)

    def transform(self, rows):
        if self.model in {"global", "signer"}:
            raise ValueError("Mean bundles use signer identifiers directly")
        matrix = self.design(rows)
        matrix -= self.mean
        matrix /= self.scale
        return matrix

    def predict(self, rows):
        """Return neutral predictions and category flags, preserving input row order/index."""
        unknown_sentence = self._inputs(rows)
        with threadpool_limits(limits=1):
            if self.model == "global":
                prediction = np.tile(self.global_mean, (len(rows), 1))
            elif self.model == "signer":
                prediction = np.array([self.signer_means[s] for s in rows.speaker])
            else:
                x = self.transform(rows)
                if self.model in {"ridge", "identifier_ridge"}:
                    prediction = x @ self.coef.T + self.intercept
                else:
                    hidden = x @ self.coefs[0] + self.intercepts[0]
                    np.maximum(hidden, 0, out=hidden)
                    prediction = hidden @ self.coefs[1] + self.intercepts[1]
        if not np.isfinite(prediction).all():
            raise ValueError("Prediction overflow: inputs produce nonfinite outputs")
        result = pd.DataFrame(prediction, columns=["predicted_inner", "predicted_outer"], index=rows.index)
        result["unknown_sentence"] = unknown_sentence
        result["unknown_signer"] = False
        return result


def save_bundle(path, preprocessor, estimator, model, context):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Bundle already exists; choose a fresh path")
    state = {"class": type(estimator).__name__}
    if model in {"global", "signer"}:
        state["parameters"] = estimator.parameters()
    else:
        state["parameters"] = estimator.get_params(deep=False)
        if model == "mlp":
            state.update(coefs=[a.tolist() for a in estimator.coefs_],
                         intercepts=[a.tolist() for a in estimator.intercepts_],
                         out_activation=estimator.out_activation_, n_iter=int(estimator.n_iter_),
                         loss_curve=[float(v) for v in estimator.loss_curve_])
        else:
            state.update(coef=estimator.coef_.tolist(), intercept=estimator.intercept_.tolist())
    required = [*POSE_COLUMNS, *CATEGORIES] if model in {"ridge", "mlp"} else list(CATEGORIES)
    payload = {"schema_version": SCHEMA_VERSION, "saved_at_oslo": now(), "model": model,
               "required_input_columns": required, "target_order": list(TARGETS), "target_scaling": False,
               "unknown_signer_policy": "reject", "unknown_sentence_policy": "zero indicators before scaling; flag",
               "runtime": runtime_versions(), "context": context,
               "preprocessing": preprocessor.state(), "estimator": state,
               "scope": "Fold-specific neutral-measurement model; known signers; released OpenFace units/radians only"}
    # Normalize tuple-valued sklearn parameters to the on-disk JSON representation.
    payload = json.loads(json.dumps(payload, allow_nan=False))
    bundle = OpenFaceBundle(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return bundle


def load_bundle(path, expected_sha256=None, require_runtime=True):
    path = Path(path)
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError("Bundle SHA-256 differs from the recorded artifact")
    try:
        payload = json.loads(path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON value: {v}")))
        return OpenFaceBundle(payload, require_runtime=require_runtime)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("Invalid OpenFace bundle JSON") from error


def predict_csv(bundle_path, input_path, output_path, expected_sha256=None):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Prediction output already exists; choose a fresh path")
    # Inspect the header before pandas can rename duplicate columns automatically.
    import csv
    with Path(input_path).open(newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate prediction input columns are not supported")
    rows = pd.read_csv(input_path, dtype={"speaker": str, "sentence": str})
    result = load_bundle(bundle_path, expected_sha256).predict(rows)
    result.insert(0, "input_position", np.arange(len(result)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        result.to_csv(stream, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    result = predict_csv(args.bundle, args.input, args.output, args.expected_sha256)
    print(f"Predicted {len(result)} rows; unknown sentences: {int(result.unknown_sentence.sum())}; saved {args.output}")


if __name__ == "__main__":
    main()
