"""Pure 3D geometry primitives.

Functions in this module are stateless: they take raw arrays / dicts in and
return raw arrays / dicts out. Reusable building blocks for calibration,
reconstruction, and epipolar reasoning.
"""

import numpy as np


def triangulate(points_2d, projection_matrices):
    """Linear (DLT) triangulation of a 3D point from N >= 2 camera views.

    points_2d:           (N, 2) array of 2D points, one per camera
    projection_matrices: (N, 3, 4) array of projection matrices

    Returns the homogeneous 3D point (4,), normalised so the last entry is 1.
    """
    points_2d = np.asarray(points_2d, dtype=float)
    projection_matrices = np.asarray(projection_matrices, dtype=float)

    # Build the 2N x 4 system from the standard DLT projection constraints
    rows = []
    for (x, y), P in zip(points_2d, projection_matrices):
        rows.append(x * P[2, :] - P[0, :])
        rows.append(y * P[2, :] - P[1, :])
    A = np.stack(rows, axis=0)

    # Solution is the right singular vector with the smallest singular value
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    return X / X[-1]


def clip_ray_to_aabb(origin, direction, aabb):
    """Clip a 3D ray to an axis-aligned bounding box (AABB) using the slab method.

    The ray is parameterised as ``point(t) = origin + t * direction``. For each
    world axis we find the t-values where the ray enters and exits the box's
    slab on that axis; the ray is inside the box when it's inside all three
    slabs simultaneously, so t_near = max(per-axis enters) and
    t_far = min(per-axis exits). If t_near > t_far the ray misses the box.

    origin, direction: array-likes with 3 entries.
    aabb: dict with keys 'x', 'y', 'z' each mapping to [min, max].

    Returns (point_near, point_far) as numpy arrays, or (None, None) if the
    ray misses the box.
    """
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)

    t_enters: list[float] = []
    t_exits: list[float] = []
    for axis_idx, axis_name in enumerate(("x", "y", "z")):
        a_min, a_max = aabb[axis_name]
        d = direction[axis_idx]
        if abs(d) < 1e-12:
            # Ray is parallel to this axis: it's either always or never inside
            # the slab. If origin is outside, the ray misses the box.
            if origin[axis_idx] < a_min or origin[axis_idx] > a_max:
                return None, None
            continue  # this axis doesn't constrain t
        t1 = (a_min - origin[axis_idx]) / d
        t2 = (a_max - origin[axis_idx]) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_enters.append(t1)
        t_exits.append(t2)

    if not t_enters:
        # Ray parallel to every axis (degenerate): no segment to draw.
        return None, None

    t_near = max(t_enters)
    t_far = min(t_exits)
    if t_near > t_far:
        return None, None  # ray misses the box

    return origin + t_near * direction, origin + t_far * direction
