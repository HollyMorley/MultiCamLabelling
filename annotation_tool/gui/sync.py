"""Timestamp synchronization and frame matching across multiple cameras."""

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from annotation_tool import paths


def zero_timestamps(timestamps):
    """Normalize timestamps so the first value is zero."""
    timestamps["Timestamp"] = timestamps["Timestamp"] - timestamps["Timestamp"][0]
    return timestamps


def adjust_timestamps(side_timestamps, other_timestamps):
    """Correct drift between the side camera and another camera's timestamps
    using linear regression on single-frame intervals."""
    mask = other_timestamps["Timestamp"].diff() < 4.045e+6
    other_single = other_timestamps[mask]
    side_single = side_timestamps[mask]
    diff = other_single["Timestamp"] - side_single["Timestamp"]

    # First pass: straighten the diff curve
    model = LinearRegression().fit(
        side_single["Timestamp"].values.reshape(-1, 1), diff.values
    )
    slope = model.coef_[0]
    intercept = model.intercept_
    straightened_diff = diff - (slope * side_single["Timestamp"] + intercept)
    correct_diff_idx = np.where(straightened_diff < straightened_diff.mean())

    # Refit on the lower half (drops outlier intervals)
    model_true = LinearRegression().fit(
        side_single["Timestamp"].values[correct_diff_idx].reshape(-1, 1),
        diff.values[correct_diff_idx],
    )
    slope_true = model_true.coef_[0]
    intercept_true = model_true.intercept_
    adjusted = other_timestamps["Timestamp"] - (
        slope_true * other_timestamps["Timestamp"] + intercept_true
    )
    return adjusted


def load_synced_video_captures(project, recording):
    """Open cv2 captures for every view and build the matched-frame list
    using timestamp synchronisation against project.reference_view.

    Returns:
        caps: dict[view -> cv2.VideoCapture]
        total_frames: int — frame count of the reference view
        matched_frames: list of per-view frame-number tuples, ordered by
            project.views (-1 marks views with no matching frame)
    """
    views = project.views
    ref = project.reference_view

    caps = {
        v: cv2.VideoCapture(paths.video_path(project, recording, v)) for v in views
    }
    total_frames = int(caps[ref].get(cv2.CAP_PROP_FRAME_COUNT))

    timestamps = {}
    for v in views:
        ts_path = paths.timestamps_path(project, recording, v)
        timestamps[v] = zero_timestamps(paths.load_timestamps_csv(ts_path))
    ts_adj = {ref: timestamps[ref]["Timestamp"].astype(float)}
    for v in views:
        if v != ref:
            ts_adj[v] = adjust_timestamps(timestamps[ref], timestamps[v])

    matched_frames = match_frames_by_timestamp(ts_adj, ref, views)
    return caps, total_frames, matched_frames


def match_frames_by_timestamp(timestamps_by_view, reference_view, views_order):
    """Match frames across N cameras using nearest-timestamp merge.

    timestamps_by_view: dict mapping view name -> Series of adjusted timestamps.
    reference_view: which view anchors the merge (each other view is merged
        into the reference's timeline).
    views_order: list of views defining the order of frame numbers in each
        returned tuple.

    Returns a list of tuples (one entry per view in views_order) with the
    matched frame number for that view, or -1 when no frame matched within
    tolerance.
    """
    buffer_ns = int(4.04e+6)

    dfs = {}
    for view in views_order:
        ts = timestamps_by_view[view].sort_values().reset_index(drop=True)
        dfs[view] = pd.DataFrame({
            "Timestamp": ts,
            f"Frame_number_{view}": range(len(ts)),
        })

    matched = dfs[reference_view]
    for view in views_order:
        if view == reference_view:
            continue
        matched = pd.merge_asof(
            matched, dfs[view], on="Timestamp",
            direction="nearest", tolerance=buffer_ns,
        )

    frame_cols = [f"Frame_number_{v}" for v in views_order]
    matched_frames = (
        matched[frame_cols]
        .map(lambda x: int(x) if pd.notnull(x) else -1)
        .values.tolist()
    )
    return matched_frames
