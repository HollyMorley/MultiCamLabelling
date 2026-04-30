"""Tests for annotation_tool/gui/sync.py — match_frames_by_timestamp.
"""

import pytest

pd = pytest.importorskip("pandas", reason="pandas required")

from annotation_tool.gui.sync import match_frames_by_timestamp

# Step / tolerance chosen so consecutive frames sit well inside tolerance,
# while LARGE_OFFSET sits well outside it.
STEP_NS = 1_000_000           # 1 ms between successive frames within a view
TOLERANCE_NS = 4_000_000      # 4 ms — accept consecutive single-frame jumps
LARGE_OFFSET = 10_000_000     # 10 ms — well outside tolerance


def _series(start, n, step=STEP_NS):
    """Helper: build a Series of monotonically increasing timestamps."""
    return pd.Series([start + i * step for i in range(n)], dtype=float)


# --- Ideal case: identical timelines line up frame-for-frame ---

def test_perfectly_aligned_three_views():
    ts = {v: _series(0, 5) for v in ["side", "front", "overhead"]}
    matched = match_frames_by_timestamp(
        ts, "side", ["side", "front", "overhead"], TOLERANCE_NS,
    )
    assert matched == [[i, i, i] for i in range(5)]


# --- Edge case: a dropped frame is bridged by nearest-neighbour matching ---

def test_dropped_frame_bridged_by_nearest_match():
    """Front drops a frame; side and overhead stay continuous. Front
    re-syncs via nearest-timestamp matching, so its frame numbers skip
    over the dropped frame — they don't stay 1:1 with side's after the drop."""
    side_ts     = pd.Series([0, 1, 2, 3, 4], dtype=float) * STEP_NS
    front_ts    = pd.Series([0, 1,    3, 4], dtype=float) * STEP_NS
    overhead_ts = pd.Series([0, 1, 2, 3, 4], dtype=float) * STEP_NS

    matched = match_frames_by_timestamp(
        {"side": side_ts, "front": front_ts, "overhead": overhead_ts},
        "side", ["side", "front", "overhead"], TOLERANCE_NS,
    )

    # Frames 0, 1 align directly across all three views. From frame 3
    # onwards, side and overhead stay aligned, but front's frame number
    # is one less than side's because of the drop.
    assert matched[0] == [0, 0, 0]
    assert matched[1] == [1, 1, 1]
    assert matched[3] == [3, 2, 3]
    assert matched[4] == [4, 3, 4]


# --- Generality: works with any number of views, not hardcoded to 3 ---

def test_more_than_three_views():
    """A 5-camera setup matches frame-for-frame the same way 3 views do.
    Confirms the function isn't implicitly hardcoded to 3 cameras."""
    views = ["side", "front", "overhead", "rear", "bottom"]
    ts = {v: _series(0, 4) for v in views}
    matched = match_frames_by_timestamp(ts, "side", views, TOLERANCE_NS)
    assert matched == [[i, i, i, i, i] for i in range(4)]
