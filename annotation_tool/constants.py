"""Hardcoded constants — UI ranges and defaults shown in the Create Project
dialog. Anything that varies per project lives in <project>/project.yaml,
not here."""

# Marker Size
DEFAULT_MARKER_SIZE = 1
MIN_MARKER_SIZE = 0.1
MAX_MARKER_SIZE = 5
MARKER_SIZE_STEP = 0.1

# Contrast and Brightness
DEFAULT_CONTRAST = 1.0
DEFAULT_BRIGHTNESS = 1.0
MIN_CONTRAST = 0.5
MAX_CONTRAST = 3.0
CONTRAST_STEP = 0.1
MIN_BRIGHTNESS = 0.5
MAX_BRIGHTNESS = 3.0
BRIGHTNESS_STEP = 0.1

# Defaults shown in the Create Project dialog. Users override these
# per-project; they are stored in project.yaml once chosen.
DEFAULT_VIEWS = ["side", "front", "overhead"]
DEFAULT_REFERENCE_VIEW = "side"
DEFAULT_NAME = ""  # default for the experimenters name in the Create Project dialog

# Filename used when saving / loading body-part labels for a view.
LABELS_CSV_BASENAME = "CollectedData"
