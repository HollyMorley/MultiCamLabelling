"""Help text for project / field-level concepts.

Single source of truth - used by:
  - the Create Project dialog (inline hints + "?" popups)
  - any future field-editing UIs

Each entry has:
  short - one-line hint shown under the field in the GUI
  long  - multi-paragraph explanation shown in the "?" popup
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldHelp:
    """Help text for a single form field.

    Both fields default to "" so an entry can supply just the inline hint
    (no "?" popup) or just the popup (no inline hint), as suits the field.
    """
    short: str = ""
    long: str = ""


FIELD_HELP: dict[str, FieldHelp] = {

    "project_name": FieldHelp(
        short="Short identifier - used as the project folder name.",
        long=(
            "A name for this project. Used as the directory name created "
            "under the chosen project root, and recorded in project.yaml.\n\n"
            "Pick something short and filesystem-safe (no spaces or slashes)."
        ),
    ),

    "project_root": FieldHelp(
        short="Parent folder where the project directory will be created.",
        long=(
            "The project will be created at <project_root>/<project_name>/."
        ),
    ),

    "views": FieldHelp(
        short="Comma-separated camera names, e.g. 'side, front, overhead'.",
        long=(
            "One name per camera. These names must appear as a segment in "
            "your video filenames (e.g. 'Demo_session1_side.avi' contains "
            "'side') - Add Videos can then auto-detect which view a "
            "picked file belongs to.\n\n"
        ),
    ),

    "reference_view": FieldHelp(
        short="Which camera anchors timestamp synchronisation.",
        long=(
            "In the case of dropped frames, multi-camera frame matching is "
            "done by linearly fitting each non-reference view's timestamps "
            "to the reference view's. The reference view's timeline is taken "
            "as ground truth.\n\n"
            "Pick the camera with the most reliable clock. If unsure, pick "
            "whichever camera's recording you most trust to be continuous and "
            "complete. If you are confident there is no frame dropping, "
            "pick any camera."
        ),
    ),

    "num_cameras": FieldHelp(
        short="Sanity check - must equal the number of views."
    ),

    "framerate_fps": FieldHelp(
        short="Camera frame rate in fps (e.g. 247).",
        long=(
            "The frame rate at which all cameras record. \n\n"
            "All cameras must record at the same frame rate."
        ),
    ),

    "intrinsics": FieldHelp(
        short="Per-camera intrinsic specs (focal length, sensor, principal "
              "point, crop offset). Edit in project.yaml.",
        long=(
            "One block per camera in cameras.views. Each block defines the "
            "physical optics and sensor geometry needed to build the camera "
            "matrix:\n"
            "  focal_length_mm        - lens focal length\n"
            "  pixel_size_x/y_mm      - physical size of one pixel\n"
            "  x/y_size_px            - sensor dimensions\n"
            "  principal_point_x/y_px - where the optical axis hits the sensor\n"
            "  crop_offset_x/y        - shift if your videos are crops of a "
            "larger sensor\n\n"
            "Required by Calibrate. Best edited directly in project.yaml — "
            "see the commented template for the expected schema."
        ),
    ),

    "world_origin_label": FieldHelp(
        short="Which calibration label sits at (0, 0, 0) in the world.",
        long=(
            "The calibration landmark that defines the origin of the world "
            "coordinate system. Its entry in calibration_label_coordinates "
            "must be [0, 0, 0]; every other label's coordinates are measured "
            "relative to this point."
        ),
    ),

    "calibration_label_coordinates": FieldHelp(
        short="Real-world (x, y, z) mm position of each calibration label. "
              "Edit in project.yaml.",
        long=(
            "Maps each calibration_label to its physical position in the "
            "world coordinate system, in millimetres. world_origin_label "
            "must appear here at [0, 0, 0]; all other labels are positioned "
            "relative to it.\n\n"
            "Required by Calibrate (used by solvePnP to recover camera pose). "
            "Best edited directly in project.yaml."
        ),
    ),

    "imaging_area": FieldHelp(
        short="Bounding box (mm) of the volume your subjects move through. "
              "Edit in project.yaml.",
        long=(
            "The 3D box that encloses every body-part position you care "
            "about, declared as per-axis [min, max] ranges. Used by Label "
            "to clip epipolar projection lines: the back-projected ray from "
            "a clicked 2D point is intersected with this box, and the "
            "entry/exit points are re-projected into the other views as a "
            "short, meaningful segment.\n\n"
            "Pick generously — anything outside this box gets cut off. "
            "Required by Label."
        ),
    ),

    "name": FieldHelp(
        short="Must match 'Experimenter' name in your DeepLabCut projects.",
        long=(
            "Recorded as 'scorer' in the saved label files (both the column "
            "header and the filename suffix: CollectedData_<name>.csv) so "
            "DeepLabCut can pick them up directly. If left blank, the "
            "column header is literally 'scorer'."
        ),
    ),

    "calibration_labels": FieldHelp(
        short="Define landmarks you'll label in Calibrate. One label per "
              "line. Must span all 3 world axes.",
        long=(
            "Here you define the real-world landmarks you will manually "
            "annotate in each camera view during the Calibrate step. The tool "
            "uses them to estimate each camera's pose (where it is and "
            "which way it's pointing).\n\n"
            "CRITICAL: pick landmarks whose 3D positions span all three "
            "world axes (x, y, and z). If they're roughly co-planar, the "
            "calibration solver is underdetermined and the resulting 3D "
            "reconstruction will be unstable or wrong. An ideal setup uses "
            "6+ well-distributed landmarks - e.g. corners of a known "
            "platform plus a couple of points clearly above or below it.\n\n"
            "One label per line."
        ),
    ),

    "body_part_labels": FieldHelp(
        short="Define bodyparts to annotate in the Label step. "
              "One label per line.",
        long=(
            "The body parts you will manually annotate on each extracted "
            "frame. Order matters - it determines the cycling order in "
            "the labelling tool, which affects how fast you can label. "
            "The first entry is the default selected label when the tool "
            "opens.\n\n"
            "Calibration_labels should also be included here and "
            "are handled specially - they get pre-populated from the "
            "calibration data. Calibration labels whose position is not "
            "fixed across frames (e.g. a door) should additionally be "
            "listed under movable_calibration_labels so they can be "
            "adjusted per frame during labelling.\n\n"
            "One label per line."
        ),
    ),

    "optimisation_reference_labels": FieldHelp(
        short="Subset of body parts used by 'Optimize Calibration'. "
              "One label per line.",
        long=(
            "After initial calibration from your clicked landmarks, the "
            "Optimize Calibration step in the Label tool refines each "
            "camera's pose to minimise the reprojection error of these "
            "reference body parts across views.\n\n"
            "Pick stable, easy-to-identify body parts that you can label "
            "reliably and that are visible in most or all views - e.g. "
            "joints rather than amorphous body sections, distinct points rather"
            "than midline approximations.\n\n"
            "Must be a subset of body_part_labels. One label per line."
        ),
    ),

    "movable_calibration_labels": FieldHelp(
        short="Optional. Calibration labels whose position changes per "
              "frame (e.g. a door). One label per line.",
        long=(
            "Most calibration landmarks are fixed structures - once "
            "labelled in the Calibrate step, the tool propagates their "
            "positions across every frame in the Label step. Any "
            "calibration label that *moves* between frames (e.g. a door "
            "that opens and closes) belongs here.\n\n"
            "Labels listed here are still used for camera pose estimation "
            "during Calibrate, but during Label they behave like body "
            "parts - placeable, draggable, and not auto-propagated.\n\n"
            "Must be a subset of calibration_labels. Optional - leave "
            "empty if every calibration landmark is fixed."
        ),
    ),

    "reference_label_weights": FieldHelp(
        short="Per-label multiplier in the Optimize Calibration cost. "
              "Format: one 'label: weight' pair per line.",
        long=(
            "Optional weights applied to each optimisation_reference_label "
            "during Optimize Calibration. Default for any unlisted label "
            "is 1.0.\n\n"
            "Increase the weight of body parts you trust most (clearer "
            "view, fewer occlusions, easier to localise). Decrease for "
            "body parts that are noisier but still informative.\n\n"
            "Format: one 'label: weight' pair per line."
        ),
    ),

}
