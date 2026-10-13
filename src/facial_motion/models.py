"""Small pitch-only regressors with training-only scaling and zero-pitch anchoring."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.gam.api import BSplines, GLMGam


def balanced_samples(recordings, maximum=150):
    """Each neutral recording contributes equally; temporal order is preserved."""
    samples = []
    for frame in recordings:
        usable = frame.loc[frame.valid_geometry & frame.inner_brow_height.notna() & frame.relative_pitch_deg.notna()].copy()
        reference = usable.loc[usable.reference_frame, "inner_brow_height"]
        if reference.empty:
            raise ValueError("Training recording lacks a valid reference")
        usable["target"] = usable.inner_brow_height - reference.median()
        samples.append(usable)
    if not samples or any(s.empty for s in samples):
        raise ValueError("No usable training samples")
    count = min(maximum, *(len(s) for s in samples))
    if count < 12:
        raise ValueError("Too few balanced training observations for the declared models")
    return pd.concat([s.iloc[np.linspace(0, len(s)-1, count).astype(int)] for s in samples], ignore_index=True)


class CalibrationModel:
    def __init__(self, method, penalty, seed=42, basis_df=6, max_iter=3000):
        if method not in {"gam", "mlp"}:
            raise ValueError("Expected gam or mlp")
        self.method, self.penalty, self.seed = method, penalty, seed
        self.basis_df, self.max_iter = basis_df, max_iter
        self.x_scaler, self.y_scaler = StandardScaler(), StandardScaler()

    def fit(self, pitch, target):
        pitch, target = np.asarray(pitch, float).reshape(-1, 1), np.asarray(target, float).reshape(-1, 1)
        if len(pitch) != len(target) or not np.isfinite(pitch).all() or not np.isfinite(target).all():
            raise ValueError("Fit expects matching finite arrays")
        self.low, self.high = float(pitch.min()), float(pitch.max())
        if self.high - self.low < 1e-6:
            raise ValueError("Insufficient training pitch variation")
        x, y = self.x_scaler.fit_transform(pitch), self.y_scaler.fit_transform(target).ravel()
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            if self.method == "mlp":
                self.estimator = MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs",
                    alpha=self.penalty, random_state=self.seed, early_stopping=False,
                    max_iter=self.max_iter, max_fun=30000)
                self.estimator.fit(x, y)
            else:
                zero = float(self.x_scaler.transform([[0]])[0, 0])
                self.spline = BSplines(x, df=[self.basis_df], degree=[3],
                    knot_kwds=[{"lower_bound": min(float(x.min()), zero), "upper_bound": max(float(x.max()), zero)}])
                self.estimator = GLMGam(y, exog=np.ones((len(y), 1)), smoother=self.spline,
                                        alpha=[self.penalty], family=sm.families.Gaussian()).fit()
            self.warnings = [str(w.message) for w in recorded]
            if any(issubclass(w.category, ConvergenceWarning) for w in recorded):
                raise ValueError("MLP failed to converge: " + "; ".join(self.warnings))
            if self.method == "gam" and not self.estimator.converged:
                raise ValueError("GAM failed to converge")
        self.anchor = float(self._predict([0])[0])
        return self

    def _predict(self, pitch):
        x = self.x_scaler.transform(np.asarray(pitch).reshape(-1, 1))
        if self.method == "gam":
            prediction = self.estimator.predict(exog=np.ones((len(x), 1)), exog_smooth=x)
        else:
            prediction = self.estimator.predict(x)
        return self.y_scaler.inverse_transform(np.asarray(prediction).reshape(-1, 1)).ravel()

    def correction(self, pitch):
        pitch = np.asarray(pitch, float)
        supported = np.isfinite(pitch) & (pitch >= self.low) & (pitch <= self.high)
        result = np.full(pitch.shape, np.nan)
        if supported.any():
            result[supported] = self._predict(pitch[supported]) - self.anchor
        return result, supported

    def apply(self, pitch, raw):
        raw, pitch = np.asarray(raw, float), np.asarray(pitch, float)
        correction, support = self.correction(pitch)
        valid = np.isfinite(raw) & np.isfinite(pitch)
        support &= valid
        corrected = np.full(raw.shape, np.nan)
        corrected[valid] = raw[valid]  # Declared raw fallback outside training support.
        corrected[support] -= correction[support]
        return corrected, support
