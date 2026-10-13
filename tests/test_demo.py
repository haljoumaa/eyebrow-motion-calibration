"""Preview safeguards that do not require a graphics/inference service."""

import numpy as np
import pytest

from facial_motion.demo import _draw_trace, video_preview


def test_existing_preview_cache_rejected_before_writes(tmp_path, monkeypatch):
    import facial_motion.demo as demo

    source = tmp_path / "source.mp4"
    source.write_bytes(b"input")
    cache = tmp_path / "preview_cache"
    cache.mkdir()
    sentinel = cache / "frozen.txt"
    sentinel.write_bytes(b"preserve")
    monkeypatch.setattr(demo, "load_config", lambda path: {})
    with pytest.raises(FileExistsError, match="cache exists"):
        video_preview(source, tmp_path / "preview.mp4", (0, .2), root=tmp_path)
    assert sentinel.read_bytes() == b"preserve"
    assert list(cache.iterdir()) == [sentinel]
    assert not (tmp_path / "preview.mp4").exists()


def test_trace_preserves_a_missing_interval():
    canvas = np.zeros((100, 140, 3), dtype=np.uint8)
    values = np.array([0., 1., np.nan, np.nan, 1., 0.])
    _draw_trace(canvas, values, values, 5, "raw", (10, 10, 110, 90))
    assert canvas[:, 10:32].any()
    assert canvas[:, 88:113].any()
    assert not canvas[:, 34:86].any()
