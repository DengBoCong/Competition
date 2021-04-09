from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import multiprocessing as mt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from typing import Any
from typing import AnyStr
from typing import Dict
from typing import TextIO
from typing import Tuple

MAX_SENTENCE_LEN = 20  # 最大句子长度


def _parse_dataset_item(example: tf.train.Example.FromString) -> tf.io.parse_single_example:
    """ 用于Dataset中的TFRecord序列化字符串恢复

    :param example: 序列化字符串
    :return: 恢复后的数据
    """
    features = {
        "first": tf.io.FixedLenFeature([MAX_SENTENCE_LEN], tf.int64,
                                       default_value=tf.zeros([MAX_SENTENCE_LEN], dtype=tf.int64)),
        "second": tf.io.FixedLenFeature([MAX_SENTENCE_LEN], tf.int64,
                                        default_value=tf.zeros([MAX_SENTENCE_LEN], dtype=tf.int64)),
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    example = tf.io.parse_single_example(serialized=example, features=features)
    return example["first"], example["second"], example["label"]


def load_dataset(record_path: AnyStr, batch_size: Any, buffer_size: Any,
                 num_parallel_reads: Any = None, data_type: AnyStr = "train",
                 reshuffle_each_iteration: Any = True, drop_remainder: Any = True) -> tf.data.Dataset:
    """ 获取Dataset

    :param record_path:
    :param batch_size: batch大小
    :param buffer_size: 缓冲大小
    :param num_parallel_reads: 读取线程数
    :param data_type: 加载数据类型，train/valid
    :param reshuffle_each_iteration: 是否每个epoch打乱
    :param drop_remainder: 是否去除余数
    :return: 加载的Dataset
    """
    if not os.path.exists(record_path):
        raise FileNotFoundError("TFRecord文件不存在，请检查后重试")

    dataset = tf.data.TFRecordDataset(filenames=record_path, num_parallel_reads=num_parallel_reads)
    dataset = dataset.map(map_func=_parse_dataset_item, num_parallel_calls=mt.cpu_count())
    if data_type == "train":
        dataset = dataset.shuffle(
            buffer_size=buffer_size, reshuffle_each_iteration=reshuffle_each_iteration
        ).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    return dataset


def random_mask(text_ids):
    """随机mask
    """
    input_ids, output_ids = [], []
    rands = np.random.random(len(text_ids))
    for r, i in zip(rands, text_ids):
        if r < 0.15 * 0.8:
            input_ids.append(4)
            output_ids.append(i)
        elif r < 0.15 * 0.9:
            input_ids.append(i)
            output_ids.append(i)
        elif r < 0.15:
            input_ids.append(np.random.choice(21962) + 7)
            output_ids.append(i)
        else:
            input_ids.append(i)
            output_ids.append(0)
    return input_ids, output_ids


def sample_convert(first_query, second_query, label):
    """转换为MLM格式
    """
    if np.random.random() < 0.5:
        first_query, second_query = second_query, first_query

    first_query, first_output = random_mask(first_query)
    second_query, second_output = random_mask(second_query)

    queries = [2] + first_query + [3] + second_query + [3]
    segments = [0] * len(queries)
    outputs = [label + 5] + first_output + [0] + second_output + [0]
    return queries, segments, outputs


def load_raw_dataset(data_path: AnyStr, max_sentence_length: Any, batch_size: Any, buffer_size: Any,
                     data_type: AnyStr, num_parallel_reads: Any = None, reshuffle_each_iteration: Any = True,
                     max_data_size: Any = 0, pair_size: Any = 3) -> tf.data.Dataset:
    """ 处理原始数据，并将处理后的数据保存为TFRecord格式，两个句子合并

    :param data_path: 原始数据路径
    :param max_sentence_length: 最大输入数据
    :param batch_size: batch大小
    :param buffer_size: 缓冲大小
    :param data_type: 加载数据类型，train/valid
    :param num_parallel_reads: 读取线程数
    :param reshuffle_each_iteration: 是否每个epoch打乱
    :param max_data_size: 最大处理数据量
    :param pair_size: 数据对大小，用于剔除不符合要求数据
    :return: 无返回值
    """
    first_queries = []
    second_queries = []
    labels = []
    count = 0

    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().strip("\n").split("\t")
            if line == "" or len(line) != pair_size:
                continue

            first = [int(num) + 7 for num in line[0].split(" ")]
            second = [int(num) + 7 for num in line[1].split(" ")]

            if len(first) == 0 or len(second) == 0:
                continue

            first_queries.append(first)
            second_queries.append(second)
            labels.append(int(line[2]) if pair_size == 3 else 0)

            count += 1
            if count % 100 == 0:
                print("\r已读取 {} 条query-pairs".format(count), end="", flush=True)
            if count == max_data_size:
                break

    def generator():
        for index, (first_query, second_query, label) in enumerate(zip(first_queries, second_queries, labels)):
            queries, segments, outputs = sample_convert(first_query, second_query, label)

            queries = tf.keras.preprocessing.sequence.pad_sequences([queries], max_sentence_length, padding="post")
            segments = tf.keras.preprocessing.sequence.pad_sequences([segments], max_sentence_length, padding="post")
            outputs = tf.keras.preprocessing.sequence.pad_sequences([outputs], max_sentence_length, padding="post")

            yield tf.squeeze(queries, axis=0), tf.squeeze(segments, axis=0), tf.squeeze(outputs, axis=0), label

    dataset = tf.data.Dataset.from_generator(generator=generator, output_signature=(
        tf.TensorSpec(shape=(max_sentence_length,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_sentence_length,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_sentence_length,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ))

    if data_type == "train":
        dataset = dataset.shuffle(
            buffer_size=buffer_size, reshuffle_each_iteration=reshuffle_each_iteration
        ).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    print("数据加载完毕，正在训练中")

    return dataset


def load_pair_dataset(data_path: AnyStr, max_sentence_length: Any, batch_size: Any, buffer_size: Any,
                      data_type: AnyStr, reshuffle_each_iteration: Any = True,
                      max_data_size: Any = 0, pair_size: Any = 3) -> tf.data.Dataset:
    """ 直接加在句子对dataset

    :param data_path: 原始数据路径
    :param max_sentence_length: 最大输入数据
    :param batch_size: batch大小
    :param buffer_size: 缓冲大小
    :param data_type: 加载数据类型，train/valid
    :param reshuffle_each_iteration: 是否每个epoch打乱
    :param max_data_size: 最大处理数据量
    :param pair_size: 数据对大小，用于剔除不符合要求数据
    :return: 无返回值
    """
    first_queries = []
    second_queries = []
    labels = []
    count = 0

    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().strip("\n").split("\t")
            if line == "" or len(line) != pair_size:
                continue

            first = [int(num) for num in line[0].split(" ")]
            second = [int(num) for num in line[1].split(" ")]

            if len(first) <= 1 or len(second) <= 0:
                continue

            first_queries.append(first)
            second_queries.append(second)
            labels.append(int(line[2]) if pair_size == 3 else 0)

            count += 1
            if count % 100 == 0:
                print("\r已读取 {} 条query-pairs".format(count), end="", flush=True)
            if count == max_data_size:
                break

    first_inputs = tf.keras.preprocessing.sequence.pad_sequences(first_queries, max_sentence_length, padding="post")
    second_inputs = tf.keras.preprocessing.sequence.pad_sequences(second_queries, max_sentence_length, padding="post")

    dataset = tf.data.Dataset.from_tensor_slices((first_inputs, second_inputs, labels))

    if data_type == "train":
        dataset = dataset.shuffle(
            buffer_size=buffer_size, reshuffle_each_iteration=reshuffle_each_iteration
        ).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    print("\n数据加载完毕，正在训练中")

    return dataset


def load_tokenizer(dict_path: AnyStr) -> Tokenizer:
    """ 加载分词器工具

    :param dict_path: 字典路径
    :return: 分词器
    """
    if not os.path.exists(dict_path):
        raise FileNotFoundError("字典不存在，请检查后重试！")

    with open(dict_path, "r", encoding="utf-8") as dict_file:
        json_string = dict_file.read().strip().strip("\n")
        tokenizer = tokenizer_from_json(json_string=json_string)

    return tokenizer


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
