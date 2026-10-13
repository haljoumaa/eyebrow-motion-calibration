import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from facial_motion.features import aspect_correct, derive_table, matrix_rotation, orientation, point_line_distances


def test_distance_preserved_by_rigid_motion_and_uniform_scale():
    eyes = np.array([[-1., 0, 0], [1., 0, 0]])
    brows = np.array([[-.5, 1., .2], [.5, 1., .2]])
    original = point_line_distances(brows, *eyes)
    rotation = Rotation.from_euler("xyz", [35, 12, -8], degrees=True).as_matrix()
    for scale in [.25, 1., 4.]:
        transformed_eyes = scale * eyes @ rotation.T + [10, 8, 3]
        transformed_brows = scale * brows @ rotation.T + [10, 8, 3]
        np.testing.assert_allclose(point_line_distances(transformed_brows, *transformed_eyes) / scale, original)


def test_aspect_and_depth_scaling():
    np.testing.assert_allclose(aspect_correct([[.5, .5, .1]], 1920, 1080), [[960, 540, 192]])


def test_rotation_removes_scale_and_reference_is_zero():
    r = Rotation.from_euler("xyz", [20, 4, 3], degrees=True).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = 3 * r
    np.testing.assert_allclose(matrix_rotation(matrix), r, atol=1e-12)
    np.testing.assert_allclose(orientation(matrix, r), [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(orientation(matrix), [20, 4, 3], atol=1e-12)


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError):
        point_line_distances([[0, 1, 0]], [0, 0, 0], [0, 0, 0])
    matrix = np.eye(4)
    matrix[0, 0] = -1
    with pytest.raises(ValueError):
        matrix_rotation(matrix)


def test_missing_detection_stays_missing():
    config = {"landmarks": {}, "orientation": {"pitch_sign": 1}}
    frame, _ = derive_table(np.full((2, 478, 3), np.nan), np.full((2, 4, 4), np.nan),
                            np.array([0., .04]), "missing", 1280, 720, config)
    assert not frame.detected.any()
    assert not frame.valid_geometry.any()
    assert frame.inner_brow_height.isna().all()
    assert frame.inner_brow_distance_px.isna().all()
