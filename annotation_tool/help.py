"""Help text for project / field-level concepts.

Single source of truth — used by:
  - the Create Project dialog (inline hints + "?" popups)
  - any future field-editing UIs

Each entry has:
  short — one-line hint shown under the field in the GUI
  long  — multi-paragraph explanation shown in the "?" popup
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
        short="Short identifier — used as the project folder name.",
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
            "'side') — Add Videos can then auto-detect which view a "
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
            "6+ well-distributed landmarks — e.g. corners of a known "
            "platform plus a couple of points clearly above or below it.\n\n"
            "One label per line."
        ),
    ),

    "body_part_labels": FieldHelp(
        short="Define bodyparts to annotate in the Label step. "
              "One label per line.",
        long=(
            "The body parts you will manually annotate on each extracted "
            "frame. Order matters — it determines the cycling order in "
            "the labelling tool, which affects how fast you can label.\n\n"
            "Calibration_labels should also be included here and "
            "are handled specially — they get pre-populated from the "
            "calibration data. Any calibration annotations which are not fully"
            "static (e.g. a door) can be adjusted during labelling.\n\n"
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
            "reliably and that are visible in most or all views — e.g. "
            "joints rather than amorphous body sections, distinct points rather"
            "than midline approximations.\n\n"
            "Must be a subset of body_part_labels. One label per line."
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
