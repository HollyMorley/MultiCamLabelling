# 3D Annotation GUI

### Multi-camera labelling tool with 3D projection assistance

A tkinter-based GUI for manual annotation of body parts across N synchronised camera views (e.g.
side, front, and overhead), with real-time 3D projection lines to guide labelling. Saved labels follow
the DeepLabCut file convention so they can be fed directly into downstream DLC training.

<p align="center">
  <img src="docs/gui_screenshot_edited.png" alt="Annotation GUI screenshot" width="100%"><br>
  <em>The labelling interface, showing the synchronised camera views and projection lines used to guide placement.</em>
</p>

<p align="center">
  <img src="docs/labelling_demo.gif" alt="3D reconstruction of a running mouse" width="100%"><br>
  <em>Downstream result: labels produced with this tool were used to train DeepLabCut models (single model per camera). These multi-view predictions are then triangulated to reconstruct 3D motion.</em>
</p>

_Note: This is a personal research tool. Future work will focus on generalising the pipeline so it can be released as a
reusable package._

## Setup

**Sample data:** a GoogleDrive folder with sample video files is available
[here](https://drive.google.com/drive/folders/1qrT0OCMl8VSDXEYe_bXN3qvMKnHLk5hS?usp=sharing).

Create the environment:

```bash
conda env create -f environment.yml
```

Run the GUI:

```bash
conda activate 3d-annotation-gui
python -m annotation_tool
```

Run tests:

```bash
pytest tests/ -v
```

## Workflow

The tool is organised around the concept of a **project**: a directory containing a `project.yaml`
config plus all the videos, extracted frames, calibration data, and labels for that experiment.

A typical session:

1. **Create or load a project** from the home screen.
2. **Add Videos** — pick one video file per camera view for a session. Each video is copied into the
   project's `Videos/` folder and registered as a *recording* in `project.yaml`.
3. **Extract** synchronised frames from a recording.
4. **Calibrate** cameras by labelling known landmarks in the extracted frames (requires definition of
   `calibration_labels` in `project.yaml`).
5. **Label** body parts on each frame, guided by 3D projection lines (requires definition of `body_part_labels`
   in `project.yaml` and a calibration to be present for the recording).

Per-recording action buttons in the project view are greyed out until their prerequisites are met.

## Project structure (codebase)

```
annotation_tool/
├── __main__.py              # Entry point (python -m annotation_tool)
├── constants.py             # UI constants and Create Project dialog defaults
├── help.py                  # Field-level help text for the Create Project dialog
├── project.py               # Project + Recording dataclasses, YAML I/O, add_recording
├── paths.py                 # Disk-layout queries, file load/save helpers
├── camera/
│   ├── calibration.py       # BasicCalibration wrapper
│   └── reconstruction.py    # CameraData, BeltPoints, DLT triangulation
└── gui/
    ├── app.py               # Top-level state machine (Home → Project → Tool → back)
    ├── home.py              # Create / Load Project landing screen
    ├── create_project.py    # Create Project dialog
    ├── project_view.py      # Recordings table with per-row Extract / Calibrate / Label
    ├── add_videos.py        # Add Videos dialog (auto-detects view from filename)
    ├── base.py              # Shared base class for tools (pan, zoom, sliders)
    ├── extract.py           # Frame extraction from synchronised videos
    ├── calibrate.py         # Camera calibration point labelling
    ├── label.py             # Body part labelling with 3D projection lines
    ├── sync.py              # Timestamp matching across cameras (for mild frame drop correction)
    └── utils.py             # Pure utility functions (scrolling, help button, colours)
tests/
├── test_project.py          # Project + paths.py tests
├── test_sync.py             # Frame matching tests
└── test_triangulation.py    # DLT triangulation test
```

## On-disk project layout

A project on disk looks like this:

```
my_project/
├── project.yaml             # Project config (cameras, label schemas, recordings list)
├── Videos/                  # Videos copied here when added via Add Videos
│   ├── Demo_session1_side.avi
│   ├── Demo_session1_side_Timestamps.csv
│   ├── Demo_session1_front.avi
│   └── ...
└── recordings/
    └── Demo_session1/       # Everything for one recording lives here
        ├── frames/
        │   ├── side/img0.png ...
        │   ├── front/
        │   └── overhead/
        ├── calibration/
        │   ├── labels.csv
        │   └── labels_enhanced.csv
        └── labels/
            ├── side/CollectedData_<name>.csv (+.h5)
            ├── front/
            └── overhead/
```

- The user's name (`name:` in `project.yaml`) is written into the saved label files as the `scorer`
  column header — matching DeepLabCut's `CollectedData_<scorer>.csv/.h5` convention so the files can
  be fed directly into DLC training.
- Camera views (`side`, `front`, `overhead` by default) are configured per-project under `cameras:`
  in `project.yaml`. Add Videos auto-detects which view a picked file corresponds to by looking for
  the view name as an underscore-separated token in the filename (e.g. `Demo_session1_side.avi`).
- Calibration labels, body-part labels, optimisation reference labels, and reference label weights
  are project-level settings in `project.yaml`. They can be left empty at project creation and
  filled in later (commented templates are included in the YAML to show where).

## Tools

### Extract Frames

_Pick synchronised video frames to be labelled._

- From the project view, click **Extract** on a recording.
- Scroll or skip through the synchronised videos and extract frame trios for labelling.
- Timestamps are used to correct for any frame misalignment across cameras.

### Calibrate Cameras

_Label known landmarks across all camera views for camera pose estimation (via OpenCV's `solvePnP`)._

- From the project view, click **Calibrate** on a recording (requires `calibration_labels` set
  in `project.yaml`).
- Scroll through the video and label the calibration points in each camera view.
- The default calibration landmarks (for the APA experiments this tool was built for) are: the
  4 corners of the first belt, the corners of the starting step edge, and the `x` sticker on
  the door — though the set is fully configurable per project.

**Controls:** Right-click to place, Shift+Right-click to delete, Left-click drag to move.

### Label Body Parts

_Label pre-defined body parts, with 3D projection estimates of each point displayed across views to guide placement._

- From the project view, click **Label** on a recording (requires extracted frames AND a saved
  calibration AND `body_part_labels` set in `project.yaml`).
- **Label View** — the camera view to label in.
- **Projection View** — the view to calculate projection lines from. If labels are present in the
  selected Projection View, projection lines (from the camera centre, crossing through the platform
  edges) are drawn on the other views to guide labelling.
- **Spacer Lines** — click once, then right-click two points on the active frame to display 12
  equally-spaced vertical guide lines along the x-axis.
- **Optimize Calibration** — adjusts the manually labelled calibration points to minimise
  reprojection error between camera views, improving the projection estimates. Requires
  `optimisation_reference_labels` and `reference_label_weights` set in `project.yaml`.
- **Save Labels** — writes one `CollectedData_<name>.csv` (and `.h5`) file per camera view under
  the recording's `labels/` folder.

**Controls:** Right-click to place, Shift+Right-click to delete, Left-click drag to move, Hover for label name.
