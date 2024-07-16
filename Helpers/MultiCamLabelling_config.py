# Paths
dir = "C:/MultiCamLabelling"
DEFAULT_CALIBRATION_FILE_PATH = "%s/CameraCalibration/default_calibration_labels.csv" %(dir)
CALIBRATION_SAVE_PATH_TEMPLATE = "%s/CameraCalibration/{video_name}/calibration_labels.csv" %(dir)
FRAME_SAVE_PATH_TEMPLATE = {
    "side": "%s/Side/{video_name}" %(dir),
    "front": "%s/Front/{video_name}" %(dir),
    "overhead": "%s/Overhead/{video_name}" %(dir),
}
LABEL_SAVE_PATH_TEMPLATE = {
    "side": "%s/Side/{video_name}" %(dir),
    "front": "%s/Front/{video_name}" %(dir),
    "overhead": "%s/Overhead/{video_name}" %(dir),
}

# Labels
CALIBRATION_LABELS = ["StartPlatL", "StepL", "StartPlatR", "StepR", "Door", "TransitionL", "TransitionR"]
BODY_PART_LABELS = ["StartPlatL", "StepL", "StartPlatR", "StepR", "Door", "TransitionL", "TransitionR",
                    "Nose", "EarL", "EarR", "Back1", "Back2", "Back3", "Back4", "Back5", "Back6", "Back7", "Back8",
                    "Back9", "Back10", "Back11", "Back12", "Tail1", "Tail2", "Tail3", "Tail4", "Tail5", "Tail6",
                    "Tail7", "Tail8", "Tail9", "Tail10", "Tail11", "Tail12",
                    "ForepawToeR", "ForepawKnuckleR", "ForepawAnkleR", "ForepawKneeR",
                    "ForepawToeL", "ForepawKnuckleL", "ForepawAnkleL", "ForepawKneeL",
                    "HindpawToeR", "HindpawKnuckleR", "HindpawAnkleR", "HindpawKneeR",
                    "HindpawToeL", "HindpawKnuckleL", "HindpawAnkleL", "HindpawKneeL"]
OPTIMIZATION_REFERENCE_LABELS = ['Nose', 'ForepawToeR', 'ForepawToeL', 'Back1', 'Back6', 'Tail12', 'StartPlatR']
REFERENCE_LABEL_WEIGHTS = {
    'Nose': 2.0,
    'ForepawToeR': 1.0,
    'ForepawToeL': 1.0,
    'Back1': 1,
    'Back6': 1,
    'Tail12': 1,
    'StartPlatR': 0.5,
}

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
