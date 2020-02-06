import numpy as np
import random
from copy import copy
from tensorflow.keras.layers import Conv2D, Add


# # Details of all possible blocks
# number_of_convs = np.array(list(range(2, 9, 2))) # [ 2  4  6  8] paired conv
# number_of_filters = 16 * (2 ** np.array(list(range(7)))) # 16 multiplier
# # [  16   32   64  128  256  512 1024]
# recursive_number = np.array(list(range(1, 6))) # [1 2 3 4]


# Test details
number_of_convs = np.array(list(range(2, 5, 2))) # [ 2  4] paired conv
number_of_filters = 16 * (2 ** np.array(list(range(4)))) # 16 multiplier
# [  16   32   64  128]
recursive_number = np.array(list(range(1, 3))) # [1 2]


class BaseBlock(object):
    def __init__(self, convs_num, filters_num, recursive_num):
        self.convs_num = convs_num
        self.filters_num = filters_num
        self.recursive_num = recursive_num
        self.kernel_size = 3 # only 3x3 filters

    # TODO equality only with one object
    def __eq__(self, other):
        if isinstance(other, BaseBlock):
            return self.convs_num == other.convs_num and \
                   self.filters_num == other.filters_num and \
                   self.recursive_num == other.recursive_num
        return False

    def __str__(self):
        s = 'C{}-F{}-R{}'.format(
            self.convs_num, self.filters_num, self.recursive_num)
        return s

    def __repr__(self):
        return self.__str__()
    
    # TODO save weights between models
    def append_to_model(self, inputs):
        # fix inputs channels
        x = Conv2D(self.filters_num, self.kernel_size, strides=1,
                   padding='same', kernel_initializer='he_normal',
                   activation='relu')(inputs)
        recursive = [x]

        for _ in range(self.convs_num // 2):
            # Add recursive before conv
            if len(recursive) > 1:
                recursive_copy = copy(recursive)
                # TODO try to use 1x1 conv on recursive connections
                add = Add()(recursive)
                recursive = recursive_copy
                # last replacement
                recursive[-1] = add
            else:
                add = recursive[-1]

            conv1 = Conv2D(self.filters_num, self.kernel_size,
                           strides=1, padding='same',
                           kernel_initializer='he_normal',
                           activation='relu')(add)
            # TODO try to use leaky relu
            conv2 = Conv2D(self.filters_num, self.kernel_size,
                           strides=1, padding='same',
                           kernel_initializer='he_normal',
                           activation='relu')(conv1)

            recursive.append(conv2)
            # remove previous recursions
            if len(recursive) > self.recursive_num:
                recursive = recursive[-self.recursive_num:]

        if len(recursive) > 1:
            return Add()(recursive)
        else:
            return recursive[-1]


def str_to_block(str_block):
    stripped = str_block.split('-')
    if stripped[0].startswith('C'):
        convs_num = int(stripped[0][1:])
    if stripped[1].startswith('F'):
        filters_num = int(stripped[1][1:])
    if stripped[2].startswith('R'):
        recursive_num = int(stripped[2][1:])

    return BaseBlock(convs_num, filters_num, recursive_num)


def generate_all_possible_blocks():
    all_blocks = []
    for conv_num in number_of_convs:
        for filter_num in number_of_filters:
            for recursive_num in recursive_number:
                # filter recursive
                if recursive_num <= conv_num / 2 + 1:
                    all_blocks.append(
                        BaseBlock(conv_num, filter_num, recursive_num))

    return all_blocks


def random_select_blocks(blocks, number_of_blocks):
    return random.choices(blocks, k=number_of_blocks)
