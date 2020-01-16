import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
from PIL import Image


model_file = './train_tmp/C4-F128-R2+C4-F128-R2+C4-F64-R1+C4-F32-R2+C2-F128-R1+C4-F32-R2+C2-F64-R2.h5'

model = tf.keras.models.load_model(model_file)
model.summary()
print(len(model.trainable_variables))
s = 0
for weight in model.trainable_variables:
    s += np.prod(weight.shape)
print(s)
# img = Image.open('lr.png').crop((0, 0, 64, 64))
# # img.save('hr_me.png')
# img = np.asarray(img, dtype=np.float32)
# img = img / 127.5 - 1
# img = np.expand_dims(img, 0)
# print(img.shape)
# pred = model.predict(img)
# pred = np.squeeze(pred)
# pred = (pred + 1) * 127.5
# pred = np.uint8(pred)
# sr_img = Image.fromarray(pred)
# sr_img.save('run_sr.png')
