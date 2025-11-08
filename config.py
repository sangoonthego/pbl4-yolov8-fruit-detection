# config.py
import torch

# Dataset paths
TRAIN_IMAGES = 'split_data/train/images'
TRAIN_LABELS = 'split_data/train/labels'
VAL_IMAGES = 'split_data/val/images'
VAL_LABELS = 'split_data/val/labels'
TEST_IMAGES = 'split_data/test/images'
TEST_LABELS = 'split_data/test/labels'

# Model/Training
NUM_CLASSES = 3
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Anchors
NUM_ANCHORS = 3

# NMS & confidence
CONF_THRESH = 0.25
NMS_THRESH = 0.45
SAVE_DIR = 'weights'

# Augmentation
MOSAIC = True
MIXUP = True
COLOR_AUG = True
HFLIP = True
MULTI_SCALE = True
