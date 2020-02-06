from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, \
                                    Concatenate, Input
from tensorflow.keras import Model


def how_many_blocks(depth):
    return 2 * depth + 1


def which_depth(num_blocks):
    return (num_blocks - 1) // 2


def UNet(input_shape, blocks):
    inputs = Input(shape=input_shape)

    depth = which_depth(len(blocks))

    x = blocks[0].append_to_model(inputs)
    block_index = 1

    skips = []
    for _ in range(depth):
        skips.append(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = blocks[block_index].append_to_model(x)
        block_index = block_index + 1
    # reverse skips
    skips = reversed(skips)
    for skip in skips:
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate()([x, skip])
        x = blocks[block_index].append_to_model(x)
        block_index = block_index + 1

    x = Conv2D(3, 1, strides=1, padding='same', activation='sigmoid')(x)

    return Model(inputs=inputs, outputs=x)
