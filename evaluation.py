import tensorflow as tf
from sr_data import SR_DATA
from loss_metric import MSE, PSNR


DATASET_NAME = 'B100'
DATASET_PATH = './DIV2K/benchmark/' + DATASET_NAME
BATCH = 32
IMG_SIZE = 64


# TODO create validation over all image (not crop)
val_dataset = SR_DATA(DATASET_PATH + '/LR_bicubic/X2/',
                      DATASET_PATH + '/HR/').dataset(
    batch_size=BATCH,
    repeat_count=2,
    random_transform=True,
    crop_size=IMG_SIZE)


def model_eval(model_path):
    model = tf.keras.models.load_model(model_path)
    model.compile(loss=MSE(), metrics=[PSNR()])
    result = model.evaluate(val_dataset, verbose=0)
    return result[1] # (loss, PSNR) -> PSNR
