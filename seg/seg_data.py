import glob
import os

import numpy as np
import skimage
from tensorflow import keras

from seg.seg_const import (
    EXPANSION_RATE,
    IMAGE_SUBDIR,
    IMG_SHAPE,
    LABEL_SUBDIR,
    LBL_SHAPE,
    LOW_MEMORY,
)

#######################
#### data pipeline ####
#######################


def get_data_filenames(parent_dir):
    image_filename_list = sorted(glob.glob(os.path.join(parent_dir, IMAGE_SUBDIR, "*")))
    label_filename_list = sorted(glob.glob(os.path.join(parent_dir, LABEL_SUBDIR, "*")))
    return image_filename_list, label_filename_list


def read_data(parent_dir):
    image_filename_list, label_filename_list = get_data_filenames(parent_dir)
    image_list = []
    label_list = []
    for imgf, labelf in zip(image_filename_list, label_filename_list):
        image_list.append(skimage.io.imread(imgf))
        label_list.append(skimage.io.imread(labelf, as_gray=True))
    return image_list, label_list


def gen_data(image, label, train=True):
    if train == True:
        # random rotation
        theta = 60 * (np.random.rand() - 0.5)
        image = skimage.transform.rotate(image, theta)
        label = skimage.transform.rotate(label, theta)
        # horizontal flip or not
        if np.random.rand() > 0.5:
            image = image[:, ::-1]
            label = label[:, ::-1]
    # resize to input shape (512,512)
    image = skimage.transform.resize(image, IMG_SHAPE)
    label = skimage.transform.resize(label, LBL_SHAPE) > 0.5
    return image, label


class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_list, label_list, batch_size=8, train=True):
        self.batch_size = batch_size
        self.image_list = image_list
        self.label_list = label_list
        self.train = train
        self.index_cnt = len(self.image_list)
        self.index_list = list(range(self.index_cnt))
        self.on_epoch_end()

    def __len__(self):
        return self.index_cnt // self.batch_size

    def __getitem__(self, k):
        ks = k * self.batch_size
        kt = ks + self.batch_size
        x = np.empty((self.batch_size, *IMG_SHAPE), dtype="float32")
        y = np.empty((self.batch_size, *LBL_SHAPE), dtype="float32")
        for i, index in enumerate(range(ks, kt)):
            real_index = self.index_list[index]
            img = self.image_list[real_index]
            lbl = self.label_list[real_index]
            if LOW_MEMORY == True:  # read from file
                img = skimage.io.imread(img)
                lbl = skimage.io.imread(lbl, as_gray=True)
            # dilation
            lbl = skimage.morphology.dilation(
                lbl, skimage.morphology.disk(EXPANSION_RATE)
            )  # dilation edge prediction for visualization
            x[i], y[i] = gen_data(img, lbl, self.train)
        return x, y

    def on_epoch_end(self):
        if self.train == True:
            np.random.shuffle(self.index_list)
