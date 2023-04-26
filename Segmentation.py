import glob
import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import functools
import sys

import skimage
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

from const import *

LOW_MEMORY = False
ROOT_DIR = r"./seg/"
TRAIN_PATH = os.path.join(ROOT_DIR, r"train/")
VALID_PATH = os.path.join(ROOT_DIR, r"valid/")
IMAGE_SUBDIR = "image"
LABEL_SUBDIR = "label"
print = functools.partial(print, flush=True)


########################
#### loss functions ####
########################


def dice_loss(y_true, y_pred, smooth=1.0):
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2.0 * intersection) / (
        tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred)) + smooth
    )


def _gaussuian_kernel_4D(kernel_size, sigma=1):
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
    _gauss = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return _gauss[..., None, None] / _gauss.sum()


def SSIM_loss(y_true, y_pred):
    c1 = 1.0
    c2 = 1.0
    p = tf.expand_dims(y_true, axis=-1)  # 4D-tensor, shape=(1,512,512,1) for Conv2D
    q = tf.expand_dims(y_pred, axis=-1)
    ker = _gaussuian_kernel_4D(kernel_size=11, sigma=1.5)
    mu_p = tf.nn.conv2d(p, ker, strides=1, padding="VALID")
    mu_q = tf.nn.conv2d(q, ker, strides=1, padding="VALID")
    mu2_p = tf.square(mu_p)
    mu2_q = tf.square(mu_q)
    mu_pq = tf.multiply(mu_p, mu_q)
    sigma2_p = tf.nn.conv2d(tf.square(p), ker, strides=1, padding="VALID") - mu2_p
    sigma2_q = tf.nn.conv2d(tf.square(q), ker, strides=1, padding="VALID") - mu2_q
    sigma_pq = tf.nn.conv2d(tf.multiply(p, q), ker, strides=1, padding="VALID") - mu_pq
    return 1.0 - tf.reduce_mean(
        (2.0 * mu_pq + c1)
        * (2.0 * sigma_pq + c2)
        / ((mu2_p + mu2_q + c1) * (sigma2_p + sigma2_q + c2))
    )


def Dice_SSIM_loss(y_true, y_pred, SSMI_Weight=1.0):
    return dice_loss(y_true, y_pred, smooth=1.0) + SSMI_Weight * SSIM_loss(
        y_true, y_pred
    )


#######################
#### data pipeline ####
#######################


def get_data_filenames(parent_dir):
    image_filename_list = sorted(
        glob.glob(os.path.join(parent_dir, IMAGE_SUBDIR, "*.png"))
    )
    label_filename_list = sorted(
        glob.glob(os.path.join(parent_dir, LABEL_SUBDIR, "*.png"))
    )
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
            x[i], y[i] = gen_data(img, lbl, self.train)
        return x, y

    def on_epoch_end(self):
        if self.train == True:
            np.random.shuffle(self.index_list)


#########################
#### network modules ####
#########################


def LeakyConv2D(x, filters, k_size=3, leaky_rate=0.1, dila=1):
    x = layers.Conv2D(filters, kernel_size=k_size, dilation_rate=dila, padding="same")(
        x
    )
    x = layers.LeakyReLU(leaky_rate)(x)
    return x


def CascadeConv2D(x, filters, conv_times, k_size=3, leaky_rate=0.1, dila=1):
    for _ in range(conv_times):
        x = LeakyConv2D(x, filters, k_size, leaky_rate, dila)
    return x


def SeparableConv2D(x, filters, dila=1, leaky_rate=0.1):
    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        dilation_rate=dila,
        use_bias=False,
    )(x)
    x = layers.LeakyReLU(leaky_rate)(x)
    x = layers.Conv2D(
        filters, kernel_size=(1, 1), strides=1, padding="same", use_bias=False
    )(x)
    x = layers.LeakyReLU(leaky_rate)(x)
    return x


def ASPP_Module(x, filters):
    b, h, w, c = x.shape
    x1 = LeakyConv2D(x, filters, k_size=1, dila=1)
    x2 = SeparableConv2D(x, filters, dila=6)
    x3 = SeparableConv2D(x, filters, dila=12)
    x4 = SeparableConv2D(x, filters, dila=18)
    x5 = layers.GlobalAveragePooling2D()(x)
    x5 = layers.Reshape((1, 1, -1))(x5)
    x5 = LeakyConv2D(x5, filters, k_size=1, dila=1)
    x5 = layers.UpSampling2D(size=(h, w), interpolation="bilinear")(x5)
    x = layers.concatenate([x1, x2, x3, x4, x5])
    x = LeakyConv2D(x, filters, k_size=1, dila=1)
    return x


def ASPP_UNet(shape, kern_size=3, filters=[64, 128, 256, 512, 1024]):
    outputShape = shape[:2]  # (512,512)
    encoders = []
    inp = layers.Input(shape)  # (512,512,3)
    depth = 0
    x = inp
    conv_times = 2
    for f in filters[:-1]:
        x = CascadeConv2D(x, f, conv_times, kern_size, leaky_rate=0.1, dila=1)
        encoders.append(x)
        x = layers.MaxPooling2D(2)(x)
        depth += 1
    x = CascadeConv2D(x, filters[-1], conv_times, kern_size, leaky_rate=0.1, dila=1)
    x = ASPP_Module(x, filters[-1])
    while depth > 0:
        depth -= 1
        f = filters[depth]
        x = layers.Conv2DTranspose(f, kernel_size=2, strides=(2, 2), padding="valid")(x)
        x = layers.Concatenate()([x, encoders.pop()])
        x = CascadeConv2D(x, f, conv_times, kern_size, leaky_rate=0.1, dila=1)
    x = LeakyConv2D(x, filters=1, k_size=1, leaky_rate=0.1, dila=1)
    x = layers.Reshape(outputShape)(x)
    model = keras.Model(inp, x, name="ASPP-UNet")
    return model


####################
#### evaluation ####
####################


def calc_recall_precision_F1score(y, py):
    _TP = np.count_nonzero(np.logical_and(y, py))
    _TN = np.count_nonzero(np.logical_and(1.0 - y, 1.0 - py))
    _FP = np.count_nonzero(np.logical_and(1.0 - y, py))
    _FN = np.count_nonzero(np.logical_and(y, 1.0 - py))
    _recall = _TP / (_TP + _FN)
    _precision = _TP / (_TP + _FP)
    _f1 = 2 * _TP / (2 * _TP + _FN + _FP)
    return _recall, _precision, _f1


def compute_avg_recall_precision_F1score(
    masks, pred_prob_map, thre=0.5, from_logits=False
):
    pred_masks = pred_prob_map.copy()
    if from_logits == True:
        pred_masks = np.exp(pred_masks) / (1.0 + np.exp(pred_masks))
    pred_masks = pred_masks > thre
    ret_list = [
        calc_recall_precision_F1score(masks[i], pred_masks[i])
        for i in range(len(masks))
    ]
    return tuple(np.array(ret_list).mean(axis=0))


def get_contour_from_raw_pred(pred_label, mask_shape, thresh=0.5):
    pred_prob_map = skimage.transform.resize(pred_label, mask_shape)
    pred_mask = pred_prob_map > thresh
    pred_mask = skimage.morphology.skeletonize(pred_mask.astype(np.uint8))
    pred_edge_img = (255.0 * pred_mask).astype(np.uint8)
    return pred_edge_img


def predict_teeth_contour(model, imgfile, resized_width=800):
    img = skimage.io.imread(imgfile)
    h, w = img.shape[:2]
    scale = resized_width / w
    rimg = skimage.transform.resize(img, IMG_SHAPE)
    raw_pred = model.predict(rimg[None, :])
    raw_pred = np.squeeze(raw_pred)
    edge_pred = get_contour_from_raw_pred(
        raw_pred, (int(scale * h), int(scale * w)), thresh=0.5
    )
    return edge_pred


def evaluate(model):
    valid_image, valid_label = get_data_filenames(VALID_PATH)
    if not LOW_MEMORY:  # sufficient memory
        valid_image, valid_label = read_data(VALID_PATH)
    valid_dg = DataGenerator(valid_image, valid_label, batch_size=1, train=False)
    valid_pred_labels = model.predict(valid_dg)
    valid_labels = np.concatenate(
        [img_lbl_pair[1] for img_lbl_pair in valid_dg], axis=0
    )
    _recall, _precision, _f1 = compute_avg_recall_precision_F1score(
        valid_labels, valid_pred_labels, thre=0.5, from_logits=False
    )
    print(
        "[Validation Data] Average Recall: {:.4f}, Average precision: {:.4f}, Average F1-score: {:.4f}".format(
            _recall, _precision, _f1
        )
    )  # F1-score = DSC


##################
#### training ####
##################


def train(
    model,
    weight_ckpt=r"./'weights-teeth-boundary-model.h5",
    batch_size=2,
    epochs=50,
    lr=0.0005,
    loss_func=Dice_SSIM_loss,
):
    model.summary()
    train_image, train_label = get_data_filenames(TRAIN_PATH)
    valid_image, valid_label = get_data_filenames(VALID_PATH)
    if not LOW_MEMORY:  # sufficient memory
        train_image, train_label = read_data(TRAIN_PATH)
        valid_image, valid_label = read_data(VALID_PATH)
        print("Load image files into memory.")
    train_dg = DataGenerator(train_image, train_label, batch_size, True)
    valid_dg = DataGenerator(valid_image, valid_label, batch_size, False)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss_func)
    model_checkpoint = ModelCheckpoint(
        weight_ckpt,
        monitor="val_loss",
        verbose=2,
        save_best_only=True,
        save_weights_only=True,
    )
    ret = model.fit(
        x=train_dg,
        validation_data=valid_dg,
        epochs=epochs,
        verbose=2,
        callbacks=[model_checkpoint],
    )
    model.load_weights(weight_ckpt)
    return model


if __name__ == "__main__":
    # Train and evaluation
    LogFile = os.path.join(ROOT_DIR, "logs", "train.log")
    # Log file
    if os.path.exists(LogFile):
        os.remove(LogFile)
    log = open(LogFile, "a", encoding="utf-8")
    sys.stdout = log

    weight_ckpt = os.path.join(ROOT_DIR, "weights", r"weights-teeth-boundary-model.h5")
    model = ASPP_UNet(IMG_SHAPE, filters=[16, 32, 64, 128, 256])

    # Train the model
    # model = train(model, weight_ckpt, batch_size=2, epochs=50, lr=0.0005, loss_func=Dice_SSIM_loss)

    model.load_weights(weight_ckpt)

    evaluate(model)
    log.close()
