import os

############################################
# Constant for Teeth boundary segmentation #
############################################


LOW_MEMORY = False  # True: to enable train and evaluation on low memory machine
ROOT_DIR = r"./seg/"
TRAIN_PATH = os.path.join(ROOT_DIR, r"train/")
VALID_PATH = os.path.join(ROOT_DIR, r"valid/")
IMAGE_SUBDIR = "image"
LABEL_SUBDIR = "label"
IMG_SHAPE = (512, 512, 3)
LBL_SHAPE = IMG_SHAPE[:2]
EXPANSION_RATE = 3  # dilation rate for teeth boundary (manually labeled teeth boundary is too thin for edge detection)
