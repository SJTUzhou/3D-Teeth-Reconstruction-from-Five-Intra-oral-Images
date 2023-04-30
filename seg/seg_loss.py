import numpy as np
import tensorflow as tf

########################
#### loss functions ####
########################


def dice_loss(y_true, y_pred, smooth=1.0):
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2.0 * intersection) / (
        tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred)) + smooth
    )


def __gaussuian_kernel_4D(kernel_size, sigma=1):
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
    _gauss = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return _gauss[..., None, None] / _gauss.sum()


def SSIM_loss(y_true, y_pred):
    c1 = 1.0
    c2 = 1.0
    p = tf.expand_dims(y_true, axis=-1)  # 4D-tensor, shape=(1,512,512,1) for Conv2D
    q = tf.expand_dims(y_pred, axis=-1)
    ker = __gaussuian_kernel_4D(kernel_size=11, sigma=1.5)
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
