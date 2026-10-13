import numpy as np
import pandas as pd
import pytest

from facial_motion.models import CalibrationModel, balanced_samples


@pytest.mark.parametrize("method,penalty", [("gam", .1), ("mlp", .01)])
def test_anchor_support_training_scalers_and_missingness(method, penalty):
    pitch = np.linspace(-20, 20, 100)
    target = .003 * pitch + .0001 * pitch**2
    model = CalibrationModel(method, penalty).fit(pitch, target)
    training_mean = model.x_scaler.mean_.copy()
    correction, supported = model.correction(np.array([0, 10, 1000, np.nan]))
    assert correction[0] == pytest.approx(0, abs=1e-12)
    assert list(supported) == [True, True, False, False]
    corrected, mask = model.apply([0, 10, 1000, np.nan], [.3, .4, .7, np.nan])
    assert corrected[0] == pytest.approx(.3)
    assert corrected[2] == .7
    assert np.isnan(corrected[3])
    np.testing.assert_array_equal(model.x_scaler.mean_, training_mean)
    assert model.x_scaler.n_samples_seen_ == 100
    if method == "mlp":
        assert sum(a.size for a in model.estimator.coefs_ + model.estimator.intercepts_) == 25


def test_balanced_recordings_and_reference_target():
    def frame(rid, n, level):
        return pd.DataFrame({"recording_id": rid, "valid_geometry": True, "inner_brow_height": level + np.arange(n)*.001,
                             "relative_pitch_deg": np.arange(n), "reference_frame": np.arange(n) < 5})
    samples = balanced_samples([frame("short", 20, .2), frame("long", 60, .9)])
    assert samples.groupby("recording_id").size().to_dict() == {"long": 20, "short": 20}
    assert samples.iloc[0].target == pytest.approx(-.002)
