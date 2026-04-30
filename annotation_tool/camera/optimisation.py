"""Refines calibration extrinsics by minimising body-part reprojection
error across views. Run after InitialCalibration.

The approach: adjust the labelled calibration landmark positions (within a
small neighbourhood of where the user placed them), re-run solvePnP for
each perturbation, and pick the perturbation that makes a chosen set of
body-part labels reproject most consistently across views.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from annotation_tool.camera.calibration import InitialCalibration
from annotation_tool.camera.geometry import (
    build_projection_matrix, project_3d_to_views, triangulate,
)


def optimize_extrinsics(
    project,
    body_part_labels_at_frame,
    calibration_points,
    intrinsics,
    extrinsics,
    boundary=5.0
):
    """Refine camera extrinsics by adjusting calibration label positions.

    Each calibration label's image position is allowed to shift by up to
    +/- boundary pixels per axis. For each candidate set of perturbed positions,
    re-run solvePnP to get fresh extrinsics, then measure body-part
    reprojection error. scipy's L-BFGS-B picks the shift that minimises it.

    project: loaded Project (views, calibration_labels, optimisation config).
    body_part_labels_at_frame: dict[label][view] -> (x, y) or None.
        The user's body-part annotations on the current frame.
    calibration_points: dict[label][view] -> (x, y) tuple or None.
        Calibration landmark image positions to perturb.
    intrinsics: dict[view] -> (3, 3) intrinsic matrix (held fixed).
    extrinsics: dict[view] -> {"rotm": (3,3), "tvec": (3,1), ...} (initial).

    Returns dict with:
        optimized_calibration_points: same shape as calibration_points,
            with refined (x, y) per (label, view).
        calibration_data: refit camera params after optimisation,
            {"intrinsics": ..., "extrinsics": ...}.
        initial_total_error, initial_errors: reprojection error before.
        final_total_error, final_errors: reprojection error after.
    """
    reference_labels, weights = project.require_optimisation_config()

    initial_total_error, initial_errors = compute_reprojection_error(
        labels=reference_labels,
        body_part_labels_at_frame=body_part_labels_at_frame,
        views=project.views,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )

    initial_flat_points = _flatten_calibration_points(
        calibration_points, project.calibration_labels, project.views,
    )
    bounds = [(p - boundary, p + boundary) for p in initial_flat_points]

    result = minimize(
        _objective_function, initial_flat_points,
        args=(
            reference_labels, weights, calibration_points,
            body_part_labels_at_frame, project, intrinsics,
        ),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 15000, "ftol": 1e-8, "gtol": 1e-5, "disp": False},
    )

    optimized_points = _reshape_calibration_points(
        result.x, calibration_points, project.calibration_labels, project.views,
    )
    new_calibration_data = estimate_extrinsics_from_labels(optimized_points, project)

    final_total_error, final_errors = compute_reprojection_error(
        labels=reference_labels,
        body_part_labels_at_frame=body_part_labels_at_frame,
        views=project.views,
        intrinsics=new_calibration_data["intrinsics"],
        extrinsics=new_calibration_data["extrinsics"],
    )

    return {
        "optimized_calibration_points": optimized_points,
        "calibration_data": new_calibration_data,
        "initial_total_error": initial_total_error,
        "initial_errors": initial_errors,
        "final_total_error": final_total_error,
        "final_errors": final_errors,
    }


def compute_reprojection_error(
    labels,
    body_part_labels_at_frame,
    views,
    intrinsics,
    extrinsics,
    weights=None,
):
    """Per-label, per-view reprojection error in pixels.

    For each label: triangulate a 3D point from its labelled image
    positions, project that 3D point back into every view, compare against
    the original labelled positions. If `weights` is given, scale each
    label's error by its entry (default 1.0).

    Returns (total_error, {label: {view: error}}).
    """
    errors = {label: {v: 0 for v in views} for label in labels}
    total_error = 0

    for label in labels:
        point_3d = triangulate_label_at_frame(
            label, body_part_labels_at_frame, views, intrinsics, extrinsics,
        )
        if point_3d is None:
            continue
        point_3d = point_3d[:3]
        projections = project_3d_to_views(point_3d, extrinsics, intrinsics, views)
        for view in views:
            pt = body_part_labels_at_frame[label][view]
            if pt is None or view not in projections:
                continue
            projected_x, projected_y = projections[view]
            original_x, original_y = pt
            error = np.sqrt(
                (projected_x - original_x) ** 2 + (projected_y - original_y) ** 2
            )
            if weights is not None:
                error *= weights.get(label, 1.0)
            errors[label][view] = error
            total_error += error
    return total_error, errors


def triangulate_label_at_frame(label, body_part_labels_at_frame, views, intrinsics, extrinsics):
    """Triangulate one label's 3D position from its labelled image
    positions across views.

    Needs at least 2 views with the label placed. Returns the homogeneous
    3D point (4,), or None if the label has fewer than 2 placements.
    """
    P_list = []
    coords = []
    for view in views:
        pt = body_part_labels_at_frame[label][view]
        if pt is not None:
            e = extrinsics[view]
            P_list.append(build_projection_matrix(intrinsics[view], e["rotm"], e["tvec"]))
            coords.append(pt)
    if len(P_list) < 2 or len(coords) < 2:
        return None
    return triangulate(np.array(coords), np.array(P_list))


def estimate_extrinsics_from_labels(calibration_points, project):
    """Re-run InitialCalibration with the given calibration label positions
    to recover fresh extrinsics. Returns {"intrinsics": ..., "extrinsics": ...}.
    """
    calibration_coordinates = pd.DataFrame([
        {
            "bodyparts": label, "coords": coord,
            **{v: calibration_points[label][v][i] for v in project.views},
        }
        for label in calibration_points
        for i, coord in enumerate(["x", "y"])
    ])
    calib = InitialCalibration(calibration_coordinates, project)
    return {
        "extrinsics": calib.estimate_cams_pose(),
        "intrinsics": calib.cameras_intrinsics,
    }


def _flatten_calibration_points(calibration_points, calibration_labels, views):
    """Pack {label: {view: (x, y)}} into a flat 1D array for scipy."""
    flat = []
    for label in calibration_labels:
        for view in views:
            if calibration_points[label][view] is not None:
                flat.extend(calibration_points[label][view])
    return np.array(flat, dtype=float)


def _reshape_calibration_points(flat_points, original, calibration_labels, views):
    """Inverse of _flatten_calibration_points.

    `original` is the pre-flatten dict, consulted only to know which
    (label, view) slots actually held a label (None slots stay None).
    """
    out = {label: {v: None for v in views} for label in calibration_labels}
    i = 0
    for label in calibration_labels:
        for view in views:
            if original[label][view] is not None:
                out[label][view] = [flat_points[i], flat_points[i + 1]]
                i += 2
    return out


def _objective_function(
    flat_points,
    reference_labels, weights, original_calibration_points,
    body_part_labels_at_frame, project, intrinsics,
):
    """Called by scipy.optimize.minimize on each iteration. Takes a candidate
    set of perturbed calibration positions (flat_points), refits the camera
    extrinsics from them, returns the weighted reprojection error.
    """
    points = _reshape_calibration_points(
        flat_points, original_calibration_points,
        project.calibration_labels, project.views,
    )
    new_data = estimate_extrinsics_from_labels(points, project)
    total_error, _ = compute_reprojection_error(
        labels=reference_labels,
        body_part_labels_at_frame=body_part_labels_at_frame,
        views=project.views,
        intrinsics=new_data["intrinsics"],
        extrinsics=new_data["extrinsics"],
        weights=weights,
    )
    return total_error
