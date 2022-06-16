import os
import glob
import random
import numpy as np
import tensorflow as tf
import sys
import skimage
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime
import functools

VERSION = "v4"
IMG_SHAPE = (512, 512, 3)
LBL_SHAPE = IMG_SHAPE[:2]
LOW_MEMORY = True
print = functools.partial(print, flush=True)

def dice_loss(y_true, y_pred, smooth=1.):
    intersection = tf.reduce_sum(y_true*y_pred)
    return 1-(2.*intersection)/(tf.reduce_sum(tf.square(y_true))+tf.reduce_sum(tf.square(y_pred))+smooth)

def _circle_kernel_4D(radius):
    _kernel = skimage.morphology.disk(radius, dtype=np.float32) - np.pad(skimage.morphology.disk(radius-1, dtype=np.float32), pad_width=1, mode='constant')
    return _kernel[...,None,None] / _kernel.sum()

Radii = [3,6,9,12,15,18]
Kernels_in = [_circle_kernel_4D(r) for r in Radii]

def HD_CV_loss(y_true, y_pred, kernels_in=Kernels_in, radii=Radii, alpha=2.0):
    p = y_true
    p_binary = tf.expand_dims(p, axis=-1) # 4D-tensor, shape=(1,512,512,1)
    q = tf.clip_by_value(y_pred, 0., 1.)
    q_binary = tf.expand_dims(tf.cast(tf.cast(q + 0.5, tf.int32), tf.float32), axis=-1) # 4D-tensor, shape=(1,512,512,1)
    _squared_diff_pq = tf.square(p-q)
    f_q_p = tf.expand_dims(tf.multiply(_squared_diff_pq, q), axis=-1) # 4D-tensor, shape=(1,512,512,1)
    f_p_q = tf.expand_dims(tf.multiply(_squared_diff_pq, p), axis=-1) # 4D-tensor, shape=(1,512,512,1)
    kernels = [tf.constant(k_in,dtype=tf.float32) for k_in in kernels_in]
    loss_hd = 0.
    for r,ker in zip(radii,kernels):
        _out1 = tf.multiply(tf.nn.conv2d(1.-p_binary, ker, strides=1, padding="SAME"), f_q_p)
        _out2 = tf.multiply(tf.nn.conv2d(p_binary, ker, strides=1, padding="SAME"), f_p_q)
        _out3 = tf.multiply(tf.nn.conv2d(1.-q_binary, ker, strides=1, padding="SAME"), f_p_q)
        _out4 = tf.multiply(tf.nn.conv2d(q_binary, ker, strides=1, padding="SAME"), f_q_p)
        loss_hd += r**alpha * tf.reduce_mean(_out1+_out2+_out3+_out4)
    return loss_hd

bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

HD_Weight = 0.01
BCE_Weight = 0.5

def Dice_HD_compound_loss(y_true, y_pred, HD_Weight=HD_Weight):
    return dice_loss(y_true, y_pred, smooth=1.) + HD_Weight * HD_CV_loss(y_true, y_pred, kernels_in=Kernels_in, radii=Radii, alpha=2.0)

def Dice_BCE_compound_loss(y_true, y_pred, BCE_Weight=BCE_Weight):
    return dice_loss(y_true, y_pred, smooth=1.) + BCE_Weight * bce_loss(y_true, y_pred)

def Dice_HD_BCE_compound_loss(y_true, y_pred, HD_Weight=HD_Weight, BCE_Weight=BCE_Weight):
    return dice_loss(y_true, y_pred, smooth=1.) + HD_Weight * HD_CV_loss(y_true, y_pred, kernels_in=Kernels_in, radii=Radii, alpha=2.0) + BCE_Weight * bce_loss(y_true, y_pred)

def read_data(parent_dir):
    file_list = glob.glob(os.path.join(parent_dir, 'image', '*.png'))
    image_list = []
    label_list = []
    for file_path in file_list:
        label_path = os.path.join(parent_dir, 'label', os.path.basename(file_path))
        image_list.append(skimage.io.imread(file_path))
        label_list.append(skimage.io.imread(label_path))
    return image_list, label_list

def get_data_filenames(parent_dir):
    image_filename_list = glob.glob(os.path.join(parent_dir, 'image', '*.png'))
    label_filename_list = [os.path.join(parent_dir, 'label', os.path.basename(image_filename)) \
                          for image_filename in image_filename_list]
    return image_filename_list, label_filename_list

def gen_data(image, label, train=True):
    if train == True:
        # random rotation
        theta = 60 * (np.random.rand() - 0.5)
        image = skimage.transform.rotate(image, theta)
        label = skimage.transform.rotate(label, theta)
        # horizontal flip or not
        if np.random.rand() > 0.5:
            image = image[:,::-1]
            label = label[:,::-1]
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
        x = np.empty((self.batch_size, *IMG_SHAPE), dtype='float32')
        y = np.empty((self.batch_size, *LBL_SHAPE), dtype='float32')
        for i, index in enumerate(range(ks, kt)):
            real_index = self.index_list[index]
            img = self.image_list[real_index]
            lbl = self.label_list[real_index]
            if LOW_MEMORY == True: # read from file
                img = skimage.io.imread(img)
                lbl = skimage.io.imread(lbl, as_gray=True)
            # dilation 
            lbl = skimage.morphology.dilation(lbl, skimage.morphology.disk(2)) # dilation edge prediction for visualization
            x[i], y[i] = gen_data(img, lbl, self.train)
        return x, y
    def on_epoch_end(self):
        if self.train == True:
            np.random.shuffle(self.index_list)


# an almost typical U-Net 2015 (different in padding)
def LeakyConv2D(x, filters, k_size=3, leaky_rate=.1, dila=1):
    x = layers.Conv2D(filters, kernel_size=k_size, dilation_rate=dila, padding='same')(x)
    x = layers.LeakyReLU(leaky_rate)(x)
    return x

def CascadeConv2D(x, filters, conv_times, k_size=3, leaky_rate=.1, dila=1):
    for _ in range(conv_times):
        x = LeakyConv2D(x, filters, k_size, leaky_rate, dila)
    return x
        
def UNet2015(shape, kern_size=3, filters=[64,128,256,512,1024]):
    outputShape = shape[:2] # (512,512)
    encoders = []
    inp = layers.Input(shape) # (512,512,3)
    depth = 0
    x = inp
    conv_times = 2
    for f in filters[:-1]:
        x = CascadeConv2D(x, f, conv_times, kern_size, leaky_rate=.1, dila=1)
        encoders.append(x)
        x = layers.MaxPooling2D(2)(x)
        depth += 1
    x = CascadeConv2D(x, filters[-1], conv_times, kern_size, leaky_rate=.1, dila=1)
    while depth > 0 :
        depth -= 1
        f = filters[depth]
        x = layers.Conv2DTranspose(f, kernel_size=2, strides=(2, 2), padding="valid")(x)
        x = layers.Concatenate()([x, encoders.pop()])
        x = CascadeConv2D(x, f, conv_times, kern_size, leaky_rate=.1, dila=1)
    x = LeakyConv2D(x, filters=1, k_size=1, leaky_rate=.1, dila=1)
    x = layers.Reshape(outputShape)(x)
    model = keras.Model(inp, x, name='UNet-2015')
    return model

def ResidualConv2D(x, filters, k_size=3, leaky_rate=.1, dila=1, batch_norm=True):
    inx = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
    for _ in range(2):
        x = layers.Conv2D(filters, kernel_size=k_size, strides=1, dilation_rate=dila, padding='same')(x)
        if batch_norm==True:
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(leaky_rate)(x)
    return layers.Add()([inx, x])

def ResUNet(shape, kern_size=3, filters=[64,128,256,512,1024]):
    outputShape = shape[:2] # (512,512)
    encoders = []
    inp = layers.Input(shape) # (512,512,3)
    depth = 0
    x = inp
    for f in filters[:-1]:
        x = ResidualConv2D(x, f, kern_size, leaky_rate=.1, dila=1)
        encoders.append(x)
        x = layers.MaxPooling2D(2)(x) + layers.AveragePooling2D(2)(x)
        depth += 1
    x = ResidualConv2D(x, filters[-1], kern_size, leaky_rate=.1, dila=1)
    while depth > 0 :
        depth -= 1
        f = filters[depth]
        x = layers.Conv2DTranspose(f, kernel_size=2, strides=(2, 2), padding="valid")(x)
        x = layers.Concatenate()([x, encoders.pop()])
        x = ResidualConv2D(x, f, kern_size, leaky_rate=.1, dila=1)
    x = LeakyConv2D(x, filters=1, k_size=1, leaky_rate=.1, dila=1)
    x = layers.Reshape(outputShape)(x)
    model = keras.Model(inp, x, name='ResUNet')
    return model



def ResConv2D(x, filters, k_size, conv_times, dial=1, leaky_rate=.1):
    inx = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
    for _ in range(conv_times):
        x = LeakyConv2D(x, filters, k_size, leaky_rate, dial)
    return layers.Add()([inx, x])

def MultiAtrousConv2D(x, filters, kern_size=3, leaky_rate=.1, dila_rates=[1,2,4,8]):
    out = []
    for dila in dila_rates:
        x = ResConv2D(x, filters, kern_size, conv_times=2, dial=dila, leaky_rate=.1)
        out.append(x)
    return layers.Concatenate()(out)

def AtrousResUNet(shape, dila_rates=[1,2,4,8], filters=[4,8,16,32,64]):
    outputShape = shape[:2] # (512,512)
    encoders = []
    inp = layers.Input(shape) # (512,512,3)
    depth = 0
    x = inp
    kern_size = 3
    for f in filters[:-1]:
        x = MultiAtrousConv2D(x, f, kern_size, leaky_rate=.1, dila_rates=dila_rates)
        encoders.append(x)
        x = layers.MaxPooling2D(2)(x) + layers.AveragePooling2D(2)(x) 
        depth += 1
    x = MultiAtrousConv2D(x, filters[-1], kern_size, leaky_rate=.1, dila_rates=dila_rates)
    while depth > 0 :
        depth -= 1
        f = filters[depth]
        x = layers.Conv2DTranspose(f, kernel_size=2, strides=(2, 2), padding="valid")(x)
        x = layers.Concatenate()([x, encoders.pop()])
        x = MultiAtrousConv2D(x, f, kern_size, leaky_rate=.1, dila_rates=dila_rates)
    x = ResConv2D(x, filters[-1], kern_size, conv_times=2, dial=1, leaky_rate=.1)
    x = LeakyConv2D(x, 1, k_size=1, leaky_rate=.1, dila=1)
    x = layers.Reshape(outputShape)(x)
    model = keras.Model(inp, x, name='AtrousResUNet')
    return model



def calc_dsc(y, py):
    _tf = np.count_nonzero(np.logical_and(y, py))
    _sum = np.count_nonzero(y)+np.count_nonzero(py)
    return 2*_tf/_sum

def compute_avg_dsc(masks, pred_prob_map, thre=0.5, from_logits=False):
    pred_masks = pred_prob_map.copy()
    if from_logits == True:
        pred_masks = np.exp(pred_masks) / (1.+np.exp(pred_masks))
    pred_masks = pred_masks > thre
    dsc_list = [calc_dsc(masks[i], pred_masks[i]) for i in range(len(masks))]
    return np.mean(dsc_list)


def save_pred_masks(pred_labels, file_names, mask_shape):
    assert len(pred_labels) == len(file_names)
    for i in range(len(file_names)):
        pred_prob_map = skimage.transform.resize(pred_labels[i], mask_shape)
        pred_prob_map = skimage.morphology.erosion(pred_prob_map, skimage.morphology.disk(2))
        pred_mask = pred_prob_map > 0.5
        pred_edge_img = (255. * pred_mask).astype(np.uint8)
        skimage.io.imsave(file_names[i], pred_edge_img)


def evaluate(model, save_pred_mask=False, mask_shape=(1080,1440)):
    valid_image, valid_label = get_data_filenames(VALID_PATH)
    if not LOW_MEMORY: # 内存充足
        valid_image, valid_label = read_data(VALID_PATH)
    valid_dg = DataGenerator(valid_image, valid_label, batch_size=1, train=False)
    # evaluate
    valid_pred_labels = model.predict(valid_dg)
    valid_labels = np.concatenate([img_lbl_pair[1] for img_lbl_pair in valid_dg], axis=0)
    valid_avg_dsc = compute_avg_dsc(valid_labels, valid_pred_labels, thre=0.5, from_logits=False)
    print("Average DICE coefficient of validation data: {:.4f}".format(valid_avg_dsc))
    if save_pred_mask == True:
        # save predicted edge mask
        valid_img_names = glob.glob(os.path.join(VALID_PATH, 'image', '*.png'))
        valid_pred_file_names = [os.path.join(VALID_PRED_PATH,os.path.basename(f)) for f in valid_img_names]
        save_pred_masks(valid_pred_labels, valid_pred_file_names, mask_shape)

        
def train():
    # model = UNet2015(IMG_SHAPE, kern_size=3, filters=[32,64,128,256]) # VERSION: v2, v3
    model = AtrousResUNet(IMG_SHAPE, dila_rates=[1,2,4,8], filters=[4,8,16,32,64])  # VERSION: v4
    print(model.summary())

    train_image, train_label = get_data_filenames(TRAIN_PATH)
    valid_image, valid_label = get_data_filenames(VALID_PATH)
    if not LOW_MEMORY: # 内存充足
        train_image, train_label = read_data(TRAIN_PATH)
        valid_image, valid_label = read_data(VALID_PATH)
    batch_size = 1
    train_dg = DataGenerator(train_image, train_label, batch_size, True)
    valid_dg = DataGenerator(valid_image, valid_label, batch_size, False)

    # train
    weight_ckpt1 = os.path.join(ROOT_DIR, r'weights-dice-5-{}.h5'.format(VERSION))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.0005), loss=dice_loss)
    model_checkpoint1 = ModelCheckpoint(weight_ckpt1, monitor='val_loss',verbose=2, save_best_only=True, save_weights_only=True)
    ret = model.fit(x=train_dg, validation_data=valid_dg, epochs=5, verbose=2, callbacks=[model_checkpoint1])
    model.load_weights(weight_ckpt1)

    weight_ckpt2 = os.path.join(ROOT_DIR, r'weights-dice-HD-45-{}.h5'.format(VERSION))
    model_checkpoint2 = ModelCheckpoint(weight_ckpt2, monitor='val_loss',verbose=2, save_best_only=True, save_weights_only=True)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.0005), loss=Dice_HD_compound_loss)
    ret = model.fit(x=train_dg, validation_data=valid_dg, epochs=45, verbose=2, callbacks=[model_checkpoint2])
    model.load_weights(weight_ckpt2)
    return model


    


if __name__ == "__main__":
    for FOLD_IDX in [5,4,3,2,1]:
        ROOT_DIR = r"./dataWithPhoto/learning/fold{}/".format(FOLD_IDX)
        TRAIN_PATH = os.path.join(ROOT_DIR, r"train/")
        VALID_PATH = os.path.join(ROOT_DIR, r"test/")
        VALID_PRED_PATH = os.path.join(VALID_PATH, r"pred-{}/".format(VERSION))
        if not os.path.exists(VALID_PRED_PATH):
            os.makedirs(VALID_PRED_PATH)
        LogFile = os.path.join(ROOT_DIR, "train-{}.log".format(VERSION))
        # Log file
        if os.path.exists(LogFile):
            os.remove(LogFile)
        log = open(LogFile, "a", encoding='utf-8')
        sys.stdout = log
        
        model = train()
        evaluate(model, save_pred_mask=True)
        log.close()