#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to model',
                    required=True)
parser.add_argument('--output', type=str, help='output model path',
                    required=True)
parser.add_argument('--log_file', type=str, help='log file',
                    required=True)
parser.add_argument('--gpu', type=int, help='on which gpu will be compute',
                    default=0)
parser.add_argument('--batch', type=int, help='training batch size',
                    default=64)
parser.add_argument('--epochs', type=int, help='training epochs',
                    default=60)
parser.add_argument('--img_size', type=int, help='img height/width',
                    default=64)
parser.add_argument('--data', type=str, help='folder with dataset',
                    default='./DIV2K')
train_args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(train_args.gpu)
import tensorflow as tf
# config gpu device
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

from sr_data import SR_DATA
from loss_metric import MSE, PSNR


def load_div2k():
    train_ds = SR_DATA(
        os.path.join(train_args.data, 'DIV2K_train_LR_bicubic', 'X2'),
        os.path.join(train_args.data, 'DIV2K_train_HR')).dataset(
            batch_size=train_args.batch,
            repeat_count=10,
            random_transform=True,
            crop_size=train_args.img_size)

    test_ds = SR_DATA(
        os.path.join(train_args.data, 'DIV2K_valid_LR_bicubic', 'X2'),
        os.path.join(train_args.data, 'DIV2K_valid_HR')).dataset(
            batch_size=train_args.batch,
            repeat_count=5,
            random_transform=True,
            crop_size=train_args.img_size)

    return train_ds, test_ds


def train(model_path):
    train_ds, test_ds = load_div2k()
    model = tf.keras.models.load_model(model_path, compile=False)

    optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.9)
    model.compile(loss=MSE(), optimizer=optimizer,
                metrics=[PSNR()])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_PSNR', patience=5, mode='max')
    csv_log = tf.keras.callbacks.CSVLogger(train_args.log_file)

    model.fit(train_ds,
                epochs=train_args.epochs,
                callbacks=[early_stop, csv_log],
                validation_data=test_ds,
                verbose=0)

    return model


def main():
    model = train(train_args.model)
    tf.keras.models.save_model(
        model, train_args.output, include_optimizer=False)


if __name__ == '__main__':
    main()
