import re
import sys
import numpy as np
import tensorflow as tf
from typing import Tuple
from typing import Any
from typing import AnyStr
from typing import Dict
from typing import NoReturn
from typing import TextIO


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def combine_mask(seq: tf.Tensor) -> Tuple:
    """对input中的不能见单位进行mask

    :param seq: 输入序列
    :param d_type: 运算精度
    :return: mask
    """
    look_ahead_mask = _create_look_ahead_mask(seq)
    padding_mask = create_padding_mask(seq)
    return tf.maximum(look_ahead_mask, padding_mask)


def create_padding_mask(seq: tf.Tensor) -> Tuple:
    """ 用于创建输入序列的扩充部分的mask

    :param seq: 输入序列
    :return: mask
    """
    seq = tf.cast(x=tf.math.equal(seq, 0), dtype=tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def _create_look_ahead_mask(seq: tf.Tensor) -> Tuple:
    """ 用于创建当前点以后位置部分的mask

    :param seq: 输入序列
    :return: mask
    """
    seq_len = tf.shape(seq)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return look_ahead_mask


def load_checkpoint(checkpoint_dir: AnyStr, execute_type: AnyStr, checkpoint_save_size: Any,
                    model: tf.keras.Model = None) -> tf.train.CheckpointManager:
    """ 加载检查点

    :param checkpoint_dir: 检查点保存目录
    :param execute_type: 执行类型
    :param checkpoint_save_size: 检查点最大保存数量
    :param model: 传入的模型
    :return: 检查点管理器
    """
    if not model:
        raise ValueError("加载检查点时所传入模型有误，请检查后重试！")

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_dir,
                                                    max_to_keep=checkpoint_save_size)

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    elif execute_type != "train" and execute_type != "preprocess":
        raise ValueError("没有检查点，请先执行train模式")

    return checkpoint_manager


class ProgressBar(object):
    """ 进度条工具 """

    EXECUTE = "%(current)d/%(total)d %(bar)s (%(percent)3d%%) %(metrics)s"
    DONE = "%(current)d/%(total)d %(bar)s - %(time).4fs/step %(metrics)s"

    def __init__(self, total: Any = 100, num: Any = 1, width: Any = 30, fmt: AnyStr = EXECUTE,
                 symbol: AnyStr = "=", remain: AnyStr = ".", output: TextIO = sys.stderr):
        """
        :param total: 执行总的次数
        :param num: 每执行一次任务数量级
        :param width: 进度条符号数量
        :param fmt: 进度条格式
        :param symbol: 进度条完成符号
        :param remain: 进度条未完成符号
        :param output: 错误输出
        """
        assert len(symbol) == 1
        self.args = {}
        self.metrics = ""
        self.total = total
        self.num = num
        self.width = width
        self.symbol = symbol
        self.remain = remain
        self.output = output
        self.fmt = re.sub(r"(?P<name>%\(.+?\))d", r"\g<name>%dd" % len(str(total)), fmt)

    def __call__(self, current: Any, metrics: AnyStr):
        """
        :param current: 已执行次数
        :param metrics: 附加在进度条后的指标字符串
        """
        self.metrics = metrics
        percent = current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + ">" + self.remain * (self.width - size - 1) + "]"

        self.args = {
            "total": self.total * self.num,
            "bar": bar,
            "current": current * self.num,
            "percent": percent * 100,
            "metrics": metrics
        }
        print("\r" + self.fmt % self.args, file=self.output, end="")

    def reset(self, total: Any, num: Any, width: Any = 30, fmt: AnyStr = EXECUTE,
              symbol: AnyStr = "=", remain: AnyStr = ".", output: TextIO = sys.stderr):
        """重置内部属性

        :param total: 执行总的次数
        :param num: 每执行一次任务数量级
        :param width: 进度条符号数量
        :param fmt: 进度条格式
        :param symbol: 进度条完成符号
        :param remain: 进度条未完成符号
        :param output: 错误输出
        """
        self.__init__(total=total, num=num, width=width, fmt=fmt,
                      symbol=symbol, remain=remain, output=output)

    def done(self, step_time: Any, fmt: AnyStr = DONE):
        """
        :param step_time: 该时间步执行完所用时间
        :param fmt: 执行完成之后进度条格式
        """
        self.args["bar"] = "[" + self.symbol * self.width + "]"
        self.args["time"] = step_time
        print("\r" + fmt % self.args + "\n", file=self.output, end="")


def get_dict_string(data: Dict, prefix: AnyStr = "- ", precision: AnyStr = ": {:.4f} "):
    """将字典数据转换成key——value字符串

    :param data: 字典数据
    :param prefix: 组合前缀
    :param precision: key——value打印精度
    :return: 字符串
    """
    result = ""
    for key, value in data.items():
        result += (prefix + key + precision).format(value)

    return result


def read_npy_file(filename):
    """
    专门用于匹配dataset的map读取文件的方法
    :param filename: 传入的文件名张量
    :return: 返回读取的数据
    """
    data = np.load(filename.numpy().decode())
    return data.astype(np.float32)


def process_train_pairs(train_enc: Any, train_dec: Any, month_enc: Any, month_dec: Any, labels: Any):
    [train_e, ] = tf.py_function(read_npy_file, [train_enc], [tf.float32, ])
    [train_d, ] = tf.py_function(read_npy_file, [train_dec], [tf.float32, ])
    [month_e, ] = tf.py_function(read_npy_file, [month_enc], [tf.float32, ])
    [month_d, ] = tf.py_function(read_npy_file, [month_dec], [tf.float32, ])
    [label_s, ] = tf.py_function(read_npy_file, [labels], [tf.float32, ])

    return train_e, train_d, month_e, month_d, label_s
