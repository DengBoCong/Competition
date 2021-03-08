import numpy as np
import tensorflow as tf
from typing import Tuple


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def create_padding_mask(seq: tf.Tensor) -> Tuple:
    """ 用于创建输入序列的扩充部分的mask

    :param seq: 输入序列
    :return: mask
    """
    seq = tf.cast(x=tf.math.equal(seq, 0), dtype=tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
