from tensorflow import keras
from tensorflow.keras import layers

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
