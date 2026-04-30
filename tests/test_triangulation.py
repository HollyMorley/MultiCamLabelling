"""Test for the DLT triangulation routine.

Project a known 3D point through two pinhole cameras and check that
triangulate() recovers the original point.
"""

import pytest


def _project(P, X):
    """Helper: project 3D point X (3,) with projection matrix P (3, 4),
    to get 2D image coords (u, v).
    """
    import numpy as np
    Xh = np.append(X, 1.0)  # homogeneous coords: (x,y,z) → (x,y,z,1)
    xh = P @ Xh  # matrix multiply: 3x4 @ 4-vec → 3-vec
    return xh[:2] / xh[2]  # dehomogenise: divide by w to get (u, v)


def test_two_views_recover_original_point():
    np = pytest.importorskip("numpy")
    from annotation_tool.camera.geometry import triangulate

    # Each P is a 3x4 projection matrix laid out as:
    #
    #         ┌── K (intrinsics) ──┐ ┌─ t  ─┐
    #         │  fx    0    cx     │ │  tx  │
    #     P = │   0   fy    cy     │ │  ty  │
    #         │   0    0     1     │ │  tz  │
    #
    # fx, fy = focal lengths (1000 here); (cx, cy) = image centre (320, 240);
    # the right column t' encodes the camera's translation. Rotation R is
    # implicit identity, so both cameras face straight down +z.
    #
    # P1: t' = (0, 0, 0)     — camera at the world origin.
    # P2: t' = (-500, 0, 0)  — second camera offset along x for stereo.
    P1 = np.array([[1000, 0, 320, 0], [0, 1000, 240, 0], [0, 0, 1, 0]],
                  dtype=float)
    P2 = np.array([[1000, 0, 320, -500], [0, 1000, 240, 0], [0, 0, 1, 0]],
                  dtype=float)

    # The 3D point we'll try to recover.
    X_true = np.array([1.0, 2.0, 10.0])

    # Project X_true into both cameras to get its 2D image coords, then
    # ask triangulate() to recover the 3D point from those 2D points
    # plus the projection matrices.
    X_est = triangulate(
        [_project(P1, X_true), _project(P2, X_true)],  # the 2D observations
        [P1, P2],  # the camera matrices
    )

    # `pytest.approx` allows a small numerical tolerance to accommodate
    # small rounding errors in the triangulation.
    assert X_est[:3] == pytest.approx(X_true, rel=1e-6)
