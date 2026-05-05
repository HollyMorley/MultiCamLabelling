"""Tests for annotation_tool/project.py and annotation_tool/paths.py.
"""

import os
import pytest

from annotation_tool import paths
from annotation_tool.project import Project


VIEWS = ["side", "front", "overhead"]


def _make_fake_videos(tmp_path, name="Demo_session1", views=VIEWS):
    """Helper: create dummy .avi + _Timestamps.csv files. Returns dict[view -> path]."""
    raw = tmp_path / "raw"
    raw.mkdir()
    paths_by_view = {}
    for view in views:
        avi = raw / f"{name}_{view}.avi"
        avi.write_bytes(b"fake")
        ts = raw / f"{name}_{view}_Timestamps.csv"
        ts.write_text("Timestamp\n0\n")
        paths_by_view[view] = str(avi)
    return paths_by_view


# --- 1. Basic creation: project on disk has the expected directory tree ---

def test_create_writes_yaml_and_dirs(tmp_path):
    p = Project.create(
        dir=str(tmp_path / "p"), project_name="p",
        views=VIEWS, reference_view="side",
    )
    assert os.path.isfile(os.path.join(p.dir, "project.yaml"))
    assert os.path.isdir(p.videos_dir())
    assert os.path.isdir(p.recordings_dir())


# --- 2. Save / load round-trip: a saved Project loads back with the same fields ---

def test_load_roundtrip(tmp_path):
    proj_dir = str(tmp_path / "p")
    Project.create(
        dir=proj_dir, project_name="p1",
        views=VIEWS, reference_view="side",
        name="Holly", calibration_labels=["A", "B"],
    )
    p = Project.load(proj_dir)
    assert p.project_name == "p1"
    assert p.name == "Holly"
    assert p.calibration_labels == ["A", "B"]


# --- 3. Validation: bad input raises a clear exception ---

def test_create_rejects_reference_view_not_in_views(tmp_path):
    with pytest.raises(ValueError):
        Project.create(
            dir=str(tmp_path / "p"), project_name="p",
            views=["side", "front"], reference_view="overhead",
        )


# --- 4. add_recording does the file-side work and registers the recording ---

def test_add_recording_copies_videos_and_persists(tmp_path):
    p = Project.create(
        dir=str(tmp_path / "p"), project_name="p",
        views=VIEWS, reference_view="side",
    )
    src = _make_fake_videos(tmp_path)
    p.add_recording("session1", src)

    # Videos and timestamp siblings copied into Videos/.
    for view in VIEWS:
        assert os.path.isfile(paths.video_path(p, p.recordings[0], view))
        assert os.path.isfile(paths.timestamps_path(p, p.recordings[0], view))

    # And the recording survives a reload — the YAML was actually written.
    p2 = Project.load(p.dir)
    assert len(p2.recordings) == 1
    assert p2.recordings[0].name == "session1"


# --- 5. Filename parsing ---

def test_parse_video_filename_detects_view():
    name, view = paths.parse_video_filename("Demo_session1_side.avi", VIEWS)
    assert view == "side"
    assert name == "Demo_session1"
