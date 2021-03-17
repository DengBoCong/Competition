from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import *


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model: Any, warmup_steps: Any = 4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step: Any):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps
        }


def loss_func_mask(real: tf.Tensor, pred: tf.Tensor, weights: tf.Tensor = None):
    """ 屏蔽填充的SparseCategoricalCrossentropy损失

    真实标签real中有0填充部分，这部分不记入预测损失

    :param weights: 样本权重
    :param real: 真实标签张量
    :param pred: logits张量
    :return: 损失平均值
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # 填充位为0，掩蔽

    loss_ = loss_object(real, pred, sample_weight=weights)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)
