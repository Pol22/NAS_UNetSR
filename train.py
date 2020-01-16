#!/usr/bin/env python

from train_args import train_args
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(train_args.gpu)
import tensorflow as tf
# config gpu device
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from div2k import DIV2K
from loss_metric import MSE, PSNR


# TODO change DIV2K to SR_DATA
def load_div2k():
    train_div2k = DIV2K(subset='train',
                        images_dir='./DIV2K/',
                        caches_dir='./DIV2K/caches')
    test_div2k = DIV2K(subset='valid',
                    images_dir='./DIV2K/',
                    caches_dir='./DIV2K/caches')

    train_ds = train_div2k.dataset(batch_size=train_args.batch,
                                repeat_count=10,
                                random_transform=True,
                                crop_size=train_args.img_size)
    test_ds = test_div2k.dataset(batch_size=train_args.batch,
                                repeat_count=5,
                                random_transform=True,
                                crop_size=train_args.img_size)

    return train_ds, test_ds


def train(model_path, gpu):
    strategy = tf.distribute.OneDeviceStrategy(device='/GPU:0')
    # train_ds, test_ds = load_div2k()
    # with tf.device('/GPU:0'):
    with strategy.scope():
        train_ds, test_ds = load_div2k()
        model = tf.keras.models.load_model(model_path)

        optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.9)
        model.compile(loss=MSE(), optimizer=optimizer,
                    metrics=[PSNR()])
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_PSNR', patience=5, mode='max')
        csv_log = tf.keras.callbacks.CSVLogger(train_args.log_file)

        model.fit(train_ds, epochs=60, callbacks=[early_stop, csv_log],
                validation_data=test_ds)
    return model


def main():
    model = train(train_args.model, train_args.gpu)
    tf.keras.models.save_model(
        model, train_args.output, include_optimizer=False)


if __name__ == '__main__':
    main()
