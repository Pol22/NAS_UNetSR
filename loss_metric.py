import tensorflow as tf


class MSE(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        diff = tf.math.squared_difference(y_true, y_pred)
        return tf.reduce_mean(diff)


class PSNR(tf.keras.metrics.Metric):
    def __init__(self, name='PSNR', **kwargs):
        super(PSNR, self).__init__(name=name, **kwargs)
        self.psnrs = self.add_weight(name='psnr', initializer='zeros')
        self.nums = self.add_weight(name='num', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        diff = tf.math.squared_difference(y_true, y_pred)
        mse = tf.reduce_mean(diff, axis=-1)
        mse = tf.reduce_mean(mse, axis=-1)
        mse = tf.reduce_mean(mse, axis=-1)
        psnr = -10.0 * tf.math.log(mse) / tf.math.log(10.0)
        self.psnrs.assign_add(tf.reduce_mean(psnr))
        self.nums.assign_add(1)

    def result(self):
        return self.psnrs / self.nums
    
    def reset_states(self):
        self.psnrs.assign(0)
        self.nums.assign(0)
