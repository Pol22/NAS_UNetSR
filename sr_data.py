import os
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE


class SR_DATA:
    def __init__(self, lr_images_dir, hr_images_dir):
        lr_images_path = os.listdir(lr_images_dir)
        hr_images_path = os.listdir(hr_images_dir)
        
        lr_images_path = map(lambda x: os.path.join(lr_images_dir, x),
                             lr_images_path)
        hr_images_path = map(lambda x: os.path.join(hr_images_dir, x),
                             hr_images_path)

        lr_images_path = sorted(lr_images_path)
        hr_images_path = sorted(hr_images_path)

        self.lr_dataset = self.images_dataset(lr_images_path)
        self.hr_dataset = self.images_dataset(hr_images_path)

    def dataset(self, batch_size=16, repeat_count=None, random_transform=True, crop_size=256):
        ds = tf.data.Dataset.zip((self.lr_dataset, self.hr_dataset))
        ds = ds.repeat(repeat_count)
        if random_transform:
            ds = ds.map(lambda lr, hr: random_crop(lr, hr, hr_crop_size=crop_size), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        
        ds = ds.map(resize_lr, num_parallel_calls=AUTOTUNE)
        ds = ds.map(normalize, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    @staticmethod
    def images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        return ds


# -----------------------------------------------------------
#  Transformations
# -----------------------------------------------------------

def normalize(lr_img, hr_img):
    lr_img = tf.cast(lr_img, tf.float32)
    hr_img = tf.cast(hr_img, tf.float32)
    # lr_img = (lr_img / 127.5) - 1
    # hr_img = (hr_img / 127.5) - 1
    lr_img = lr_img / 255.0
    hr_img = hr_img / 255.0
    return lr_img, hr_img


def resize_lr(lr_img, hr_img):
    lr_shape = tf.shape(lr_img)
    h = lr_shape[0]
    w = lr_shape[1]
    lr_img = tf.image.resize(lr_img, size=(h * 2, w * 2),
                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                             antialias=True)
    return lr_img, hr_img


def random_crop(lr_img, hr_img, hr_crop_size=29, scale=2):
    lr_crop_size = hr_crop_size // 2
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * 2
    hr_h = lr_h * 2

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)
