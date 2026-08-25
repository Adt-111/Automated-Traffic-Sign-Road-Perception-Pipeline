"""
Central configuration for the Automated Traffic Sign & Road Perception
Pipeline: paths, GTSRB class names, and hyperparameters shared across the
classical detector, CNN classifier, and Streamlit app.
"""

import os
import random

import numpy as np
import torch

# Paths — EDIT for your environment

# Expected GTSRB layout (official Kaggle "GTSRB - German Traffic Sign" dataset):
#   DATA_ROOT/Train/<class_id>/*.png
#   DATA_ROOT/Test/*.png + Test.csv (ClassId,Path columns)
DATA_ROOT = os.environ.get("GTSRB_ROOT", "./data/gtsrb")
TRAIN_DIR = os.path.join(DATA_ROOT, "Train")
TEST_DIR = os.path.join(DATA_ROOT, "Test")
TEST_CSV = os.path.join(DATA_ROOT, "Test.csv")

CHECKPOINT_DIR = "./checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "cnn_classifier.pth")
SVM_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "sift_svm.pkl")
TRAINING_CURVES_PATH = os.path.join(CHECKPOINT_DIR, "training_curves.png")
CONFUSION_MATRIX_PATH = os.path.join(CHECKPOINT_DIR, "confusion_matrix.png")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GTSRB class names (43 classes, official ordering)

GTSRB_CLASSES = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
    9: "No passing", 10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection", 12: "Priority road", 13: "Yield",
    14: "Stop", 15: "No vehicles", 16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry", 18: "General caution", 19: "Dangerous curve to the left",
    20: "Dangerous curve to the right", 21: "Double curve", 22: "Bumpy road",
    23: "Slippery road", 24: "Road narrows on the right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed and passing limits", 33: "Turn right ahead",
    34: "Turn left ahead", 35: "Ahead only", 36: "Go straight or right",
    37: "Go straight or left", 38: "Keep right", 39: "Keep left",
    40: "Roundabout mandatory", 41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}
NUM_CLASSES = len(GTSRB_CLASSES)

# Classical detector (HSV masking + Sobel/contours + SIFT-SVM baseline)

# HSV ranges tuned for typical traffic-sign colors. Red wraps around hue=0,
# so it needs two ranges. Values are OpenCV HSV convention: H in [0,179].
HSV_RANGES = {
    "red": [
        {"lower": (0, 70, 50), "upper": (10, 255, 255)},
        {"lower": (170, 70, 50), "upper": (179, 255, 255)},
    ],
    "blue": [{"lower": (100, 100, 50), "upper": (130, 255, 255)}],
    "yellow": [{"lower": (18, 80, 80), "upper": (35, 255, 255)}],
}

MIN_CONTOUR_AREA = 400            # pixels; discards tiny noise contours
MAX_CONTOUR_AREA_RATIO = 0.5      # discards contours covering most of the frame (false positives)
MIN_ASPECT_RATIO = 0.5            # bounding-box width/height sanity bounds for sign-like shapes
MAX_ASPECT_RATIO = 2.0
MORPH_KERNEL_SIZE = 5
SIFT_N_FEATURES = 200              # cap per-image keypoints for speed on the SVM baseline
BOW_VOCAB_SIZE = 100                # bag-of-visual-words cluster count for the SIFT+SVM pipeline

# CNN classifier

CNN_IMAGE_SIZE = 48                 # GTSRB signs resized to 48x48 before the CNN
CNN_BATCH_SIZE = 64
CNN_NUM_EPOCHS = 15
CNN_LEARNING_RATE = 1e-3
CNN_WEIGHT_DECAY = 1e-4
CNN_VAL_SPLIT = 0.15
CNN_DROPOUT = 0.4

IMAGENET_MEAN = [0.3403, 0.3121, 0.3214]   # approximate GTSRB channel stats
IMAGENET_STD = [0.2724, 0.2608, 0.2669]

# App / inference

CONFIDENCE_THRESHOLD = 0.5          # minimum softmax confidence to display a detection
CROP_PADDING_RATIO = 0.10           # extra margin added around each candidate bbox before classifying

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
