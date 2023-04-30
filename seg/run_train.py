import functools
import os
import sys

import numpy as np
import skimage
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from seg.seg_const import IMG_SHAPE, LOW_MEMORY, ROOT_DIR, TRAIN_PATH, VALID_PATH
from seg.seg_data import DataGenerator, get_data_filenames, read_data
from seg.seg_loss import Dice_SSIM_loss
from seg.seg_model import ASPP_UNet

print = functools.partial(print, flush=True)

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
    weight_ckpt,
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
    # Log file
    LogFile = os.path.join(ROOT_DIR, "logs", "train_evaluation.log")
    os.makedirs(os.path.dirname(LogFile), exist_ok=True)
    if os.path.exists(LogFile):
        os.remove(LogFile)
    log = open(LogFile, "a", encoding="utf-8")
    sys.stdout = log

    weight_ckpt = os.path.join(ROOT_DIR, "weights", r"model_weights.h5")
    model = ASPP_UNet(IMG_SHAPE, filters=[16, 32, 64, 128, 256])

    # Train the model
    model = train(
        model, weight_ckpt, batch_size=2, epochs=50, lr=0.0005, loss_func=Dice_SSIM_loss
    )

    model.load_weights(weight_ckpt)

    evaluate(model)
    log.close()
