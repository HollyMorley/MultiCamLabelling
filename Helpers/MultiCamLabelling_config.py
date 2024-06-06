# config.py

# Paths
DEFAULT_CALIBRATION_FILE_PATH = "X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/CameraCalibration/default_calibration_labels.csv"
CALIBRATION_SAVE_PATH_TEMPLATE = "X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/CameraCalibration/{video_name}/calibration_labels.csv"
FRAME_SAVE_PATH_TEMPLATE = {
    "side": "X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Side/{video_name}",#/img{frame_number}.png",
    "front": "X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Front/{video_name}",#/img{frame_number}.png",
    "overhead": "X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Overhead/{video_name}"#"/img{frame_number}.png"
}
LABEL_SAVE_PATH_TEMPLATE = {
    "side": "X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Side/{video_name}/CollectedData_Holly.csv",
    "front": "X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Front/{video_name}/CollectedData_Holly.csv",
    "overhead": "X:/hmorley/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Overhead/{video_name}/CollectedData_Holly.csv"
}

# Labels
CALIBRATION_LABELS = ["StartPlatL", "StepL", "StartPlatR", "StepR", "Door", "TransitionL", "TransitionR"]
BODY_PART_LABELS = ["Nose", "EarL", "EarR", "Back1", "Back2", "Tail1", "ForepawToeR", "HindpawToeL"]

# Marker Size
DEFAULT_MARKER_SIZE = 3
MIN_MARKER_SIZE = 0.5
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
