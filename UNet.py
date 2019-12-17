import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, \
    UpSampling2D, Concatenate, Input
from tensorflow.keras import Model


def downsample(filters, kernel_size=3, strides=1):
    result = tf.keras.Sequential()
    result.add(MaxPooling2D(pool_size=(2, 2)))
    result.add(Conv2D(filters, kernel_size, strides=strides, padding='same',
                      kernel_initializer='he_normal', activation='relu'))
    result.add(Conv2D(filters, kernel_size, strides=strides, padding='same',
                      kernel_initializer='he_normal', activation='relu'))

    return result


def upsample(filters, kernel_size=3, strides=1):
    result = tf.keras.Sequential()
    result.add(UpSampling2D(size=(2, 2)))
    result.add(Conv2D(filters, kernel_size, strides=strides, padding='same',
                      kernel_initializer='he_normal', activation='relu'))
    result.add(Conv2D(filters, kernel_size, strides=strides, padding='same',
                      kernel_initializer='he_normal', activation='relu'))

    return result


def start(filters, kernel_size=3, strides=1):
    result = tf.keras.Sequential()
    result.add(Conv2D(filters, kernel_size, strides=strides, padding='same',
                      kernel_initializer='he_normal', activation='relu'))
    result.add(Conv2D(filters, kernel_size, strides=strides, padding='same',
                      kernel_initializer='he_normal', activation='relu'))

    return result


def UNet(shape):
    inputs = Input(shape=shape)

    down_stack = [
        downsample(128),
        downsample(256),
        downsample(512),
        downsample(1024)
    ]

    up_stack = [
        upsample(512),
        upsample(256),
        upsample(128),
        upsample(64)
    ]

    conv1 = Conv2D(64, 3, strides=1, padding='same', kernel_initializer='he_normal')
    conv2 = Conv2D(64, 3, strides=1, padding='same', kernel_initializer='he_normal')
    x = conv1(inputs)
    x = conv2(x)

    skips = []
    for down in down_stack:
        skips.append(x)
        x = down(x)
    # reverse skips
    skips = reversed(skips)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    x = Conv2D(3, 1, strides=1, padding='same', kernel_initializer='he_normal')(x)

    return Model(inputs=inputs, outputs=x)


# unet = UNet((256, 256, 3))
# tf.keras.utils.plot_model(unet, show_shapes=True)
