# config.py

# Paths
DEFAULT_CALIBRATION_FILE_PATH = "H:/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/CameraCalibration/default_calibration_labels.csv"
CALIBRATION_SAVE_PATH_TEMPLATE = "H:/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/CameraCalibration/{video_name}/calibration_labels.csv"
FRAME_SAVE_PATH_TEMPLATE = {
    "side": "H:/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Side/{video_name}",
    "front": "H:/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Front/{video_name}",
    "overhead": "H:/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Overhead/{video_name}"
}

LABEL_SAVE_PATH_TEMPLATE = {
    "side": "H:/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Side/{video_name}",
    "front": "H:/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Front/{video_name}",
    "overhead": "H:/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Overhead/{video_name}"
}


# Labels
CALIBRATION_LABELS = ["StartPlatL", "StepL", "StartPlatR", "StepR", "Door", "TransitionL", "TransitionR"]
BODY_PART_LABELS = ["StartPlatL", "StepL", "StartPlatR", "StepR", "Door", "TransitionL", "TransitionR",
                    "Nose", "EarL", "EarR", "Back1", "Back2", "Back3", "Back4", "Back5", "Back6", "Back7", "Back8",
                    "Back9", "Back10", "Back11", "Back12", "Tail1", "Tail2", "Tail3", "Tail4", "Tail5", "Tail6",
                    "Tail7", "Tail8", "Tail9", "Tail10", "Tail11", "Tail12", "ForepawToeR", "ForepawAnkleR",
                    "ForepawShoulderR", "ForepawToeL", "ForepawAnkleL", "ForepawShoulderL", "HindpawToeR",
                    "HindpawAnkleR", "HindpawShoulderR", "HindpawToeL", "HindpawAnkleL", "HindpawShoulderL"]

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

# Coordinate data format
SCORER = "Holly"
