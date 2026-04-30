"""Project model — load and save project.yaml, manage recordings.

A project is a directory containing:

    <project_dir>/
        project.yaml
        Videos/                 # videos copied here when added via add_recording
        recordings/<name>/      # extracted frames, calibration, labels for one session

`project.yaml` describes the camera setup and label schemas, plus a list of
recordings. Each recording is one experiment session with one video per
camera view. See README / docs for the full schema.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import date

import yaml

from annotation_tool.constants import DEFAULT_NAME


PROJECT_FILE = "project.yaml"
VIDEOS_DIRNAME = "Videos"
RECORDINGS_DIRNAME = "recordings"


# Commented-out templates for fields that are optional at project creation
# but required before specific tools will run. When a field is None at save
# time, its template is appended to project.yaml so users see where to fill
# it in. After they uncomment and edit, the field becomes real on next load
# and its template is no longer emitted.
_OPTIONAL_TEMPLATES: dict[str, str] = {
    "name": (
        "# name:                             # your name; written into label files for downstream DeepLabCut analysis\n"
        "#   <your_name>"
    ),
    "calibration_labels": (
        "# calibration_labels:               # required by Calibrate. Pick landmarks spanning all 3 world axes.\n"
        "#   - <calibration_landmark_1>\n"
        "#   - <calibration_landmark_2>"
    ),
    "body_part_labels": (
        "# body_part_labels:                 # required by Label. The body parts you will annotate per frame.\n"
        "#   - <body_part_1>\n"
        "#   - <body_part_2>"
    ),
    "optimisation_reference_labels": (
        "# optimisation_reference_labels:    # required by Label > Optimize Calibration. Subset of body_part_labels.\n"
        "#   - <body_part_1>"
    ),
    "reference_label_weights": (
        "# reference_label_weights:          # required by Label > Optimize Calibration. Weight per label (default 1.0).\n"
        "#   <body_part_1>: 1.0"
    ),
    "movable_calibration_labels": (
        "# movable_calibration_labels:       # optional. Subset of calibration_labels whose position changes per frame\n"
        "#                                   # (e.g. doors). Treated like body parts during labelling.\n"
        "#   - <calibration_landmark_1>"
    ),
    "framerate_fps": (
        "# framerate_fps: <framerate>   "
        "#                              "
    ),
    "world_origin_label": (
        "# world_origin_label: <calibration_label>   # required by Calibrate. Which calibration label sits at\n"
        "#                                           # (0, 0, 0) in the world coordinate system."
    ),
    "calibration_label_coordinates": (
        "# calibration_label_coordinates:    # required by Calibrate. Real-world (x, y, z) position in mm of\n"
        "#                                   # each calibration_label. Must include world_origin_label at [0,0,0].\n"
        "#   StartPlatR: [0.0,    0.0,  0.0]\n"
        "#   StartPlatL: [0.0,   53.5,  0.0]\n"
        "#   TransitionR: [470.0,  0.0,  0.0]\n"
        "#   TransitionL: [470.0, 53.5,  0.0]\n"
        "#   Door:        [-8.2,  26.0, 48.5]\n"
        "#   StepR:       [0.0,    0.0,  5.0]\n"
        "#   StepL:       [0.0,   53.5,  5.0]"
    ),
    "imaging_area": (
        "# imaging_area:                     # required by Label. Bounding box (mm) of the volume your subjects\n"
        "#                                   # move through. Used to clip epipolar projection lines to a\n"
        "#                                   # meaningful segment.\n"
        "#   x: [0,   610]\n"
        "#   y: [0,    53.5]\n"
        "#   z: [0,    60]"
    ),
    "intrinsics": (
        "# intrinsics:                       # required by Calibrate. Per-camera intrinsic specs. One block per\n"
        "#                                   # view in cameras.views. Focal length and pixel size build the\n"
        "#                                   # pinhole matrix; crop offsets shift the principal point if your\n"
        "#                                   # videos are crops of a larger sensor.\n"
        "#   side:\n"
        "#     focal_length_mm: 16\n"
        "#     pixel_size_x_mm: 4.8e-3\n"
        "#     pixel_size_y_mm: 4.8e-3\n"
        "#     x_size_px: 1920\n"
        "#     y_size_px: 230\n"
        "#     principal_point_x_px: 960\n"
        "#     principal_point_y_px: 600\n"
        "#     crop_offset_x: 0\n"
        "#     crop_offset_y: 607\n"
        "#   front:\n"
        "#     focal_length_mm: 12\n"
        "#     pixel_size_x_mm: 3.45e-3\n"
        "#     pixel_size_y_mm: 3.45e-3\n"
        "#     x_size_px: 296\n"
        "#     y_size_px: 320\n"
        "#     principal_point_x_px: 960\n"
        "#     principal_point_y_px: 600\n"
        "#     crop_offset_x: 652\n"
        "#     crop_offset_y: 477\n"
        "#   overhead:\n"
        "#     focal_length_mm: 16\n"
        "#     pixel_size_x_mm: 4.8e-3\n"
        "#     pixel_size_y_mm: 4.8e-3\n"
        "#     x_size_px: 992\n"
        "#     y_size_px: 116\n"
        "#     principal_point_x_px: 960\n"
        "#     principal_point_y_px: 600\n"
        "#     crop_offset_x: 608\n"
        "#     crop_offset_y: 914"
    ),
}


@dataclass
class Recording:
    """One experiment session with one video per camera view.

    `videos` maps view name -> path relative to the project directory.
    Examples:
        - name: "Demo session 1"
        - videos:
            side: "Videos/Demo_session1_side.avi"
            ...
    """
    name: str
    videos: dict[str, str]


@dataclass
class Project:
    dir: str
    project_name: str
    views: list[str]
    reference_view: str
    num_cameras: int
    name: str | None = None     # experimenter name
    calibration_labels: list[str] | None = None
    body_part_labels: list[str] | None = None
    optimisation_reference_labels: list[str] | None = None
    reference_label_weights: dict[str, float] | None = None
    movable_calibration_labels: list[str] | None = None
    framerate_fps: float | None = None
    world_origin_label: str | None = None
    calibration_label_coordinates: dict[str, list[float]] | None = None
    imaging_area: dict[str, list[float]] | None = None
    intrinsics: dict[str, dict[str, float]] | None = None
    recordings: list[Recording] = field(default_factory=list)
    created: str | None = None

    # ----- Construction -----

    @classmethod
    def create(
        cls,
        dir: str,
        project_name: str,
        views: list[str],
        reference_view: str,
        num_cameras: int | None = None,
        name: str | None = DEFAULT_NAME,
        calibration_labels: list[str] | None = None,
        body_part_labels: list[str] | None = None,
        optimisation_reference_labels: list[str] | None = None,
        reference_label_weights: dict[str, float] | None = None,
        movable_calibration_labels: list[str] | None = None,
        framerate_fps: float | None = None,
        world_origin_label: str | None = None,
        calibration_label_coordinates: dict[str, list[float]] | None = None,
        imaging_area: dict[str, list[float]] | None = None,
        intrinsics: dict[str, dict[str, float]] | None = None,
    ) -> "Project":
        """Create a new project on disk: makes the directory tree and writes
        project.yaml. Raises ValueError on invalid input or FileExistsError
        if project.yaml already exists at `dir`."""
        if not project_name:
            raise ValueError("Project name cannot be empty.")
        if not views:
            raise ValueError("At least one camera view is required.")
        if reference_view not in views:
            raise ValueError(
                f"reference_view {reference_view!r} must be one of views {views}."
            )
        if num_cameras is None:
            num_cameras = len(views)
        if num_cameras != len(views):
            raise ValueError(
                f"num_cameras ({num_cameras}) must equal len(views) ({len(views)})."
            )

        os.makedirs(dir, exist_ok=True)
        project_path = os.path.join(dir, PROJECT_FILE)
        if os.path.exists(project_path):
            raise FileExistsError(
                f"A project already exists at {project_path}. "
                "Use Project.load() to open it."
            )

        project = cls(
            dir=os.path.abspath(dir),
            project_name=project_name,
            views=list(views),
            reference_view=reference_view,
            num_cameras=num_cameras,
            name=name,
            calibration_labels=calibration_labels,
            body_part_labels=body_part_labels,
            optimisation_reference_labels=optimisation_reference_labels,
            reference_label_weights=reference_label_weights,
            movable_calibration_labels=movable_calibration_labels,
            framerate_fps=framerate_fps,
            world_origin_label=world_origin_label,
            calibration_label_coordinates=calibration_label_coordinates,
            imaging_area=imaging_area,
            intrinsics=intrinsics,
            recordings=[],
            created=date.today().isoformat(),
        )

        os.makedirs(project.videos_dir(), exist_ok=True)
        os.makedirs(project.recordings_dir(), exist_ok=True)
        project.save()
        return project

    @classmethod
    def load(cls, dir: str) -> "Project":
        """Load an existing project from <dir>/project.yaml."""
        project_path = os.path.join(dir, PROJECT_FILE)
        if not os.path.exists(project_path):
            raise FileNotFoundError(
                f"No {PROJECT_FILE} found at {dir}. "
                "Choose a directory containing a project, or create one."
            )
        with open(project_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cameras = data.get("cameras", {}) or {}
        views = list(cameras.get("views", []))
        reference_view = cameras.get("reference_view")
        num_cameras = cameras.get("num_cameras", len(views))
        if num_cameras != len(views):
            raise ValueError(
                f"{PROJECT_FILE}: cameras.num_cameras ({num_cameras}) "
                f"does not match len(cameras.views) ({len(views)})."
            )
        if reference_view not in views:
            raise ValueError(
                f"{PROJECT_FILE}: cameras.reference_view {reference_view!r} "
                f"must be one of cameras.views {views}."
            )

        recordings = [
            Recording(name=r["name"], videos=dict(r["videos"]))
            for r in (data.get("recordings") or [])
        ]

        return cls(
            dir=os.path.abspath(dir),
            project_name=data.get(
                "project_name", os.path.basename(os.path.abspath(dir)),
            ),
            views=views,
            reference_view=reference_view,
            num_cameras=num_cameras,
            name=data.get("name"),
            calibration_labels=data.get("calibration_labels"),
            body_part_labels=data.get("body_part_labels"),
            optimisation_reference_labels=data.get("optimisation_reference_labels"),
            reference_label_weights=data.get("reference_label_weights"),
            movable_calibration_labels=data.get("movable_calibration_labels"),
            framerate_fps=data.get("framerate_fps"),
            world_origin_label=data.get("world_origin_label"),
            calibration_label_coordinates=data.get("calibration_label_coordinates"),
            imaging_area=data.get("imaging_area"),
            intrinsics=data.get("intrinsics"),
            recordings=recordings,
            created=data.get("created"),
        )

    def save(self) -> None:
        """Write project.yaml. Optional fields that are None are omitted from
        the active YAML, but a commented template is appended so users know
        where to fill them in."""
        data: dict = {
            "project_name": self.project_name,
        }
        if self.created:
            data["created"] = self.created
        data["cameras"] = {
            "views": list(self.views),
            "reference_view": self.reference_view,
            "num_cameras": self.num_cameras,
        }
        # Track which optional fields are set vs. missing so we can emit
        # commented templates for the missing ones.
        optional_state = {
            "name": self.name,
            "calibration_labels": self.calibration_labels,
            "body_part_labels": self.body_part_labels,
            "optimisation_reference_labels": self.optimisation_reference_labels,
            "reference_label_weights": self.reference_label_weights,
            "movable_calibration_labels": self.movable_calibration_labels,
            "framerate_fps": self.framerate_fps,
            "world_origin_label": self.world_origin_label,
            "calibration_label_coordinates": self.calibration_label_coordinates,
            "imaging_area": self.imaging_area,
            "intrinsics": self.intrinsics,
        }
        for field, value in optional_state.items():
            if value is not None:
                data[field] = (
                    list(value) if isinstance(value, list) else
                    dict(value) if isinstance(value, dict) else
                    value
                )
        data["recordings"] = [
            {"name": r.name, "videos": dict(r.videos)} for r in self.recordings
        ]

        yaml_text = yaml.safe_dump(data, sort_keys=False)

        # Build commented-template block for any missing optional fields.
        missing_templates = [
            _OPTIONAL_TEMPLATES[f]
            for f, value in optional_state.items()
            if value is None
        ]
        if missing_templates:
            template_block = (
                "\n# --- Fill in before running Calibrate / Label ---\n\n"
                + "\n\n".join(missing_templates)
                + "\n"
            )
            # Insert before the recordings: line so optional project-level
            # config groups with the rest of the project header.
            marker = "recordings:"
            idx = yaml_text.find(marker)
            if idx == -1:
                yaml_text = yaml_text + template_block
            else:
                yaml_text = yaml_text[:idx] + template_block + "\n" + yaml_text[idx:]

        with open(os.path.join(self.dir, PROJECT_FILE), "w", encoding="utf-8") as f:
            f.write(yaml_text)

    # ----- Paths -----

    def videos_dir(self) -> str:
        return os.path.join(self.dir, VIDEOS_DIRNAME)

    def recordings_dir(self) -> str:
        return os.path.join(self.dir, RECORDINGS_DIRNAME)

    # ----- Recordings -----

    def add_recording(
        self,
        name: str,
        video_paths: dict[str, str],
    ) -> Recording:
        """Copy each video into Videos/, append the recording to self.recordings,
        and write project.yaml so the change is saved to disk.

        `video_paths` maps view name -> absolute source path. All views in
        `self.views` must be present.
        """
        if any(r.name == name for r in self.recordings):
            raise ValueError(f"A recording named {name!r} already exists.")
        missing = set(self.views) - set(video_paths.keys())
        if missing:
            raise ValueError(
                f"Recording {name!r} is missing videos for views: {sorted(missing)}."
            )
        extra = set(video_paths.keys()) - set(self.views)
        if extra:
            raise ValueError(
                f"Recording {name!r} has videos for unknown views: {sorted(extra)}."
            )

        os.makedirs(self.videos_dir(), exist_ok=True)
        relative = {}
        for view, src in video_paths.items():
            src = os.path.abspath(src)
            if not os.path.isfile(src):
                raise FileNotFoundError(f"Video for view {view!r} not found: {src}")
            dst = os.path.join(self.videos_dir(), os.path.basename(src))
            if os.path.abspath(dst) != src:
                shutil.copy2(src, dst)
            relative[view] = os.path.relpath(dst, self.dir).replace(os.sep, "/")

            # Copy the sibling _Timestamps.csv if present
            ts_src = os.path.splitext(src)[0] + "_Timestamps.csv"
            if os.path.isfile(ts_src):
                ts_dst = os.path.join(self.videos_dir(), os.path.basename(ts_src))
                if os.path.abspath(ts_dst) != ts_src:
                    shutil.copy2(ts_src, ts_dst)

        recording = Recording(name=name, videos=relative)
        self.recordings.append(recording)
        self.save()
        return recording

    # ----- Required-config accessors -----

    def require_calibration_labels(self) -> list[str]:
        if not self.calibration_labels:
            raise ValueError(
                "project.yaml is missing 'calibration_labels'. "
                "Add the list of calibration landmark names and reload the project."
            )
        return self.calibration_labels

    def require_body_part_labels(self) -> list[str]:
        if not self.body_part_labels:
            raise ValueError(
                "project.yaml is missing 'body_part_labels'. "
                "Add the list of body-part names and reload the project."
            )
        return self.body_part_labels

    def require_framerate_fps(self) -> float:
        if not self.framerate_fps:
            raise ValueError(
                "project.yaml is missing 'framerate_fps'. "
                "Add the camera frame rate (Hz) and reload the project."
            )
        return float(self.framerate_fps)

    def require_intrinsics(self) -> dict[str, dict[str, float]]:
        if not self.intrinsics:
            raise ValueError(
                "project.yaml is missing 'intrinsics'. "
                "Add per-camera intrinsic specs (one block per view) "
                "and reload the project."
            )
        missing = set(self.views) - set(self.intrinsics.keys())
        if missing:
            raise ValueError(
                f"project.yaml 'intrinsics' is missing entries for views: "
                f"{sorted(missing)}."
            )
        return self.intrinsics

    def require_calibration_geometry(
        self,
    ) -> tuple[str, dict[str, list[float]], dict[str, list[float]]]:
        """Return (world_origin_label, calibration_label_coordinates,
        imaging_area), validated against each other and against
        calibration_labels.

        Raises ValueError on any missing or inconsistent field.
        """
        if not self.world_origin_label:
            raise ValueError(
                "project.yaml is missing 'world_origin_label'."
            )
        if not self.calibration_label_coordinates:
            raise ValueError(
                "project.yaml is missing 'calibration_label_coordinates'."
            )
        if not self.imaging_area:
            raise ValueError(
                "project.yaml is missing 'imaging_area'."
            )
        # Origin must be a known calibration label and sit at (0, 0, 0)
        if self.world_origin_label not in self.calibration_label_coordinates:
            raise ValueError(
                f"world_origin_label {self.world_origin_label!r} has no entry "
                f"in calibration_label_coordinates."
            )
        origin_xyz = self.calibration_label_coordinates[self.world_origin_label]
        if list(origin_xyz) != [0.0, 0.0, 0.0]:
            raise ValueError(
                f"calibration_label_coordinates[{self.world_origin_label!r}] "
                f"must be [0, 0, 0]; got {origin_xyz}."
            )
        # All calibration_labels must have coordinates
        if self.calibration_labels:
            missing = set(self.calibration_labels) - set(
                self.calibration_label_coordinates.keys()
            )
            if missing:
                raise ValueError(
                    f"calibration_label_coordinates is missing entries for: "
                    f"{sorted(missing)}."
                )
        # imaging_area sanity
        for axis in ("x", "y", "z"):
            if axis not in self.imaging_area:
                raise ValueError(f"imaging_area is missing axis {axis!r}.")
            lo, hi = self.imaging_area[axis]
            if lo >= hi:
                raise ValueError(
                    f"imaging_area.{axis} must have min < max; got [{lo}, {hi}]."
                )
        return (
            self.world_origin_label,
            self.calibration_label_coordinates,
            self.imaging_area,
        )

    def require_optimisation_config(self) -> tuple[list[str], dict[str, float]]:
        if not self.optimisation_reference_labels or not self.reference_label_weights:
            raise ValueError(
                "project.yaml is missing 'optimisation_reference_labels' or "
                "'reference_label_weights'. Add both before running "
                "Optimize Calibration."
            )
        return self.optimisation_reference_labels, self.reference_label_weights

    # ----- Is each tool ready to run? Drives the greyed-out buttons. -----
    #
    # Each method below checks if tool has everything it needs.
    # Returns None if yes; otherwise a short explanation of what's missing,
    # which project_view shows as a tooltip on the greyed-out button.
    #
    # These only check things stored in project.yaml. Whether frames have
    # been extracted or calibration has been saved for a specific recording
    # is checked separately by project_view (using helpers in paths.py).

    def extract_ready(self) -> str | None:
        if not self.framerate_fps:
            return "framerate_fps not set in project.yaml"
        return None

    def calibrate_ready(self) -> str | None:
        if not self.calibration_labels:
            return "calibration_labels not set in project.yaml"
        if not self.framerate_fps:
            return "framerate_fps not set in project.yaml"
        if not self.intrinsics:
            return "intrinsics not set in project.yaml"
        if not (
            self.world_origin_label
            and self.calibration_label_coordinates
            and self.imaging_area
        ):
            return (
                "calibration geometry not set "
                "(world_origin_label / calibration_label_coordinates / imaging_area)"
            )
        return None

    def label_ready(self) -> str | None:
        if not self.body_part_labels:
            return "body_part_labels not set in project.yaml"
        return self.calibrate_ready()  # Label needs everything Calibrate needs
