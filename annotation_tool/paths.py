"""Path queries and file load/save helpers"""

import os
import pandas as pd

from annotation_tool.constants import LABELS_CSV_BASENAME
from annotation_tool.project import Project, Recording


def parse_video_filename(
    filename: str, views: list[str]) -> tuple[str, str | None]:
    """Attempt at extraction of (recording_name, view) from a video filename.

    The view is whichever underscore-separated token in the
    filename equals one of `views` (case-sensitive). The recording name is
    everything else, rejoined with underscores. Extension is dropped first.

    Returns (name, view); `view` is None if no token matched.
    """
    base, _ = os.path.splitext(os.path.basename(filename))
    parts = base.split("_")
    view = None
    for p in parts:
        if p in views:
            view = p
            break
    name_parts = [p for p in parts if p != view]
    name = "_".join(name_parts) if name_parts else base
    return name, view


def videos_initial_dir(project: Project) -> str:
    """Initial directory for file pickers — the project's videos/ folder if it
    exists, else the project root."""
    vd = project.videos_dir()
    return vd if os.path.isdir(vd) else project.dir


def recording_dir(project: Project, recording: Recording) -> str:
    return os.path.join(project.recordings_dir(), recording.name)


def video_path(project: Project, recording: Recording, view: str) -> str:
    """Absolute path to the video for `view` in `recording`."""
    rel = recording.videos[view]
    return os.path.normpath(os.path.join(project.dir, rel))


def timestamps_path(project: Project, recording: Recording, view: str) -> str:
    """Path to the timestamp CSV alongside the video. Convention: same basename
    as the video, with _Timestamps.csv suffix."""
    vp = video_path(project, recording, view)
    return os.path.splitext(vp)[0] + "_Timestamps.csv"


def labeled_data_dir(project: Project, recording: Recording, view: str) -> str:
    """Per-view directory holding both the extracted frames (PNG) and the
    body-part label files (CSV + H5). Follows the DeepLabCut convention of
    keeping each frame next to its labels."""
    return os.path.join(recording_dir(project, recording), "labeled_data", view)


def frame_image_path(
    project: Project, recording: Recording, view: str, frame_number: int
) -> str:
    return os.path.join(labeled_data_dir(project, recording, view), f"img{frame_number}.png")


def calibration_dir(project: Project, recording: Recording) -> str:
    return os.path.join(recording_dir(project, recording), "calibration")


def calibration_csv(project: Project, recording: Recording) -> str:
    return os.path.join(calibration_dir(project, recording), "labels.csv")


def calibration_csv_enhanced(project: Project, recording: Recording) -> str:
    return os.path.join(calibration_dir(project, recording), "labels_enhanced.csv")


def default_calibration_csv(project: Project) -> str:
    """Project-wide default calibration CSV. Lives at the project root so it
    can be offered as a fallback when a recording has no calibration of its own."""
    return os.path.join(project.dir, "default_calibration.csv")


def _labels_basename(scorer: str | None) -> str:
    return f"{LABELS_CSV_BASENAME}_{scorer}" if scorer else LABELS_CSV_BASENAME


def labels_csv(
    project: Project, recording: Recording, view: str,
    scorer: str | None = None,
) -> str:
    """Path to the per-view labels CSV. `scorer` is the DLC convention name
    inserted into the filename (`CollectedData_<scorer>.csv`); defaults to
    the project's `name` (the labeller)."""
    scorer = scorer if scorer is not None else project.name
    return os.path.join(labeled_data_dir(project, recording, view), f"{_labels_basename(scorer)}.csv")


def labels_h5(
    project: Project, recording: Recording, view: str,
    scorer: str | None = None,
) -> str:
    """Path to the per-view labels HDF5 sibling. See labels_csv for the
    `scorer` parameter."""
    scorer = scorer if scorer is not None else project.name
    return os.path.join(labeled_data_dir(project, recording, view), f"{_labels_basename(scorer)}.h5")


# =============================================================================
# Availability checks (drives greying-out in project_view)
# =============================================================================

def _dir_has_files(path: str, extensions: tuple[str, ...]) -> bool:
    if not os.path.isdir(path):
        return False
    return any(
        f.lower().endswith(extensions) for f in os.listdir(path)
    )


def has_extracted_frames(project: Project, recording: Recording) -> bool:
    """True if every view has at least one extracted frame on disk."""
    image_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    return all(
        _dir_has_files(labeled_data_dir(project, recording, v), image_exts)
        for v in project.views
    )


def has_calibration(project: Project, recording: Recording) -> bool:
    """True if a calibration CSV has been saved for this recording."""
    return os.path.isfile(calibration_csv(project, recording)) or os.path.isfile(
        calibration_csv_enhanced(project, recording)
    )


def has_labels(
    project: Project, recording: Recording, view: str | None = None,
) -> bool:
    """True if labels exist for the given view, or for any view when view is None."""
    views = [view] if view is not None else project.views
    return any(os.path.isfile(labels_csv(project, recording, v)) for v in views)


# =============================================================================
# Load / save helpers — encapsulate makedirs + format choice
# =============================================================================

def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_calibration_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_calibration_csv(df: pd.DataFrame, path: str) -> None:
    _ensure_parent(path)
    df.to_csv(path, index=False)


def save_frame_image(
    project: Project, recording: Recording, view: str, frame_number: int,
    img,
) -> str:
    """Write `img` (BGR ndarray, as produced by cv2) to the canonical path.
    Returns the path written."""
    import cv2
    path = frame_image_path(project, recording, view, frame_number)
    _ensure_parent(path)
    cv2.imwrite(path, img)
    return path


def load_timestamps_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_labels_h5(path: str) -> pd.DataFrame:
    return pd.read_hdf(path, key="df")


def save_labels(
    df: pd.DataFrame, project: Project, recording: Recording, view: str,
    scorer: str | None = None,
) -> tuple[str, str]:
    """Save body-part labels for one view to both .csv and .h5.
    Returns (csv_path, h5_path)."""
    csv_path = labels_csv(project, recording, view, scorer=scorer)
    h5_path = labels_h5(project, recording, view, scorer=scorer)
    _ensure_parent(csv_path)
    df.to_csv(csv_path)
    df.to_hdf(h5_path, key="df", mode="w", format="fixed")
    return csv_path, h5_path
