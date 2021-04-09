from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jieba
import tensorflow as tf
from code.tools import load_tokenizer
from typing import Any
from typing import AnyStr
from typing import NoReturn
from typing import List


def slice_data(data_path: AnyStr, first_split_path: AnyStr, second_split_path: AnyStr, split: Any = 0.1) -> NoReturn:
    """ 划分原始数据集

    :param data_path: 原始数据集路径
    :param first_split_path: 划分后的第一个数据集保存路径
    :param second_split_path: 划分后的第二个数据集保存路径
    :param split: 划分比例
    :return: 无返回值
    """
    first_count, second_count, count = 0, 0, 0
    with open(data_path, "r", encoding="utf-8") as data_file, open(
            first_split_path, "w", encoding="utf-8") as first_file, open(
        second_split_path, "w", encoding="utf-8") as second_file:
        for line in data_file:
            line = line.strip().strip("\n")
            if line == "":
                continue

            if first_count % (100 * (1 - split) + 1) == 0 and second_count % (100 * split + 1) == 0:
                first_count += 1
                second_count += 1

            if first_count % (100 * (1 - split) + 1) != 0:
                first_file.write(line + "\n")
                first_count += 1
            elif second_count % (100 * split + 1) != 0:
                second_file.write(line + "\n")
                second_count += 1

            count += 1
            if count % 1000 == 0:
                print("\r已划分 {} 条query-pairs".format(count), end="", flush=True)


def preprocess_raw_data(data_path: AnyStr, record_data_path: AnyStr, dict_path: AnyStr,
                        max_len: Any, max_data_size: Any = 0, pair_size: Any = 3) -> NoReturn:
    """ 处理原始数据，并将处理后的数据保存为TFRecord格式

    :param data_path: 原始数据路径
    :param record_data_path: 分词好的数据路径
    :param dict_path: 字典保存路径
    :param max_len: 最大序列长度
    :param max_data_size: 最大处理数据量
    :param pair_size: 数据对大小，用于剔除不符合要求数据
    :return: 无返回值
    """
    first_queries = []
    second_queries = []
    labels = []
    count = 0
    tokenizer = load_tokenizer(dict_path=dict_path)

    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().strip("\n").split("\t")
            if line == "" or len(line) != pair_size:
                continue

            first = " ".join(jieba.cut(line[0]))
            second = " ".join(jieba.cut(line[1]))

            if len(first.split(" ")) == 0 or len(second.split(" ")) == 0:
                continue

            first_queries.append(first)
            second_queries.append(second)
            labels.append(int(line[2]) if pair_size == 3 else 0)
            count += 1
            if count % 100 == 0:
                print("\r已读取 {} 条query-pairs".format(count), end="", flush=True)
            if count == max_data_size:
                break
    first_queries_seq = tokenizer.texts_to_sequences(first_queries)
    second_queries_seq = tokenizer.texts_to_sequences(second_queries)

    first_queries_seq = tf.keras.preprocessing.sequence.pad_sequences(first_queries_seq,
                                                                      maxlen=max_len, dtype="int32", padding="post")
    second_queries_seq = tf.keras.preprocessing.sequence.pad_sequences(second_queries_seq,
                                                                       maxlen=max_len, dtype="int32", padding="post")

    writer = tf.data.experimental.TFRecordWriter(record_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((first_queries_seq, second_queries_seq, labels))

    def generator():
        for first_query, second_query, label in dataset:
            example = tf.train.Example(features=tf.train.Features(feature={
                "first": tf.train.Feature(int64_list=tf.train.Int64List(value=first_query)),
                "second": tf.train.Feature(int64_list=tf.train.Int64List(value=second_query)),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))
            yield example.SerializeToString()

    print("\n正在写入数据，请稍后")
    serialized_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())
    writer.write(serialized_dataset)

    print("数据预处理完毕，TFRecord数据文件已保存！")


def preprocess_raw_data_not_tokenized(data_path: AnyStr, record_data_path: AnyStr,
                                      max_len: Any, max_data_size: Any = 0, pair_size: Any = 3) -> NoReturn:
    """ 处理原始数据，并将处理后的数据保存为TFRecord格式

    :param data_path: 原始数据路径
    :param record_data_path: 分词好的数据路径
    :param max_len: 最大序列长度
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

            first = line[0].split(" ")
            second = line[1].split(" ")

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

    first_queries_seq = tf.keras.preprocessing.sequence.pad_sequences(first_queries, maxlen=max_len,
                                                                      dtype="int32", padding="post")
    second_queries_seq = tf.keras.preprocessing.sequence.pad_sequences(second_queries, maxlen=max_len,
                                                                       dtype="int32", padding="post")

    writer = tf.data.experimental.TFRecordWriter(record_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((first_queries_seq, second_queries_seq, labels))

    def generator():
        for first_query, second_query, label in dataset:
            example = tf.train.Example(features=tf.train.Features(feature={
                "first": tf.train.Feature(int64_list=tf.train.Int64List(value=first_query)),
                "second": tf.train.Feature(int64_list=tf.train.Int64List(value=second_query)),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))
            yield example.SerializeToString()

    print("\n正在写入数据，请稍后")
    serialized_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())
    writer.write(serialized_dataset)

    print("数据预处理完毕，TFRecord数据文件已保存！")


def preprocess_raw_data_not_tokenized_combine(data_path: AnyStr, record_data_path: AnyStr,
                                              max_len: Any, max_data_size: Any = 0, pair_size: Any = 3) -> NoReturn:
    """ 处理原始数据，并将处理后的数据保存为TFRecord格式，两个句子合并

    :param data_path: 原始数据路径
    :param record_data_path: 分词好的数据路径
    :param max_len: 最大序列长度
    :param max_data_size: 最大处理数据量
    :param pair_size: 数据对大小，用于剔除不符合要求数据
    :return: 无返回值
    """
    queries = []
    segments = []
    labels = []
    count = 0

    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().strip("\n").split("\t")
            if line == "" or len(line) != pair_size:
                continue

            first = line[0].split(" ")
            second = line[1].split(" ")

            temp = [21963] + first + [21964] + second + [21964]
            temp1 = [1 for _ in range(len(first) + 2)] + [2 for _ in range(len(second) + 1)]

            if len(first) == 0 or len(second) == 0:
                continue

            queries.append(temp)
            segments.append(temp1)
            labels.append(int(line[2]) if pair_size == 3 else 0)

            count += 1
            if count % 100 == 0:
                print("\r已读取 {} 条query-pairs".format(count), end="", flush=True)
            if count == max_data_size:
                break

    queries_seq = tf.keras.preprocessing.sequence.pad_sequences(queries, maxlen=max_len,
                                                                dtype="int32", padding="post")
    segments_seq = tf.keras.preprocessing.sequence.pad_sequences(segments, maxlen=max_len,
                                                                 dtype="int32", padding="post")

    writer = tf.data.experimental.TFRecordWriter(record_data_path)
    dataset = tf.data.Dataset.from_tensor_slices((queries_seq, segments_seq, labels))

    def generator():
        for first_query, second_query, label in dataset:
            example = tf.train.Example(features=tf.train.Features(feature={
                "first": tf.train.Feature(int64_list=tf.train.Int64List(value=first_query)),
                "second": tf.train.Feature(int64_list=tf.train.Int64List(value=second_query)),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))
            yield example.SerializeToString()

    print("\n正在写入数据，请稍后")
    serialized_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())
    writer.write(serialized_dataset)

    print("数据预处理完毕，TFRecord数据文件已保存！")


def slice_neg_pos_data(data_path: AnyStr, save_path: AnyStr, if_self: bool = False) -> NoReturn:
    """ 将句子划分为正负样本集

    :param data_path: 原始数据集路径
    :param save_path: 数据增强瘦的数据保存路径
    :param if_self: 是否使用自身pairs
    :return:
    """
    remain = dict()
    res = dict()
    positive = list()
    negative = list()
    count = 0
    negative_set = set()

    def find(key: AnyStr) -> AnyStr:
        if key != remain[key]:
            remain[key] = find(remain[key])
        return remain[key]

    def union(key1: AnyStr, key2: AnyStr) -> NoReturn:
        remain[find(key2)] = find(key1)

    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().strip("\n").split("\t")
            if len(line) != 3:
                continue
            if line[2] == "1":
                if remain.get(line[0], "a") == "a":
                    remain[line[0]] = line[0]
                if remain.get(line[1], "a") == "a":
                    remain[line[1]] = line[1]
                positive.append([line[0], line[1]])
            elif line[2] == "0":
                negative.append([line[0], line[1]])
                if if_self:
                    negative_set.add(line[0])
                    negative_set.add(line[1])

        for first_query, second_query in positive:
            union(first_query, second_query)

        for first_query, second_query in positive:
            if res.get(find(first_query), "a") == "a":
                res[find(first_query)] = set()
            res[find(first_query)].add(first_query)
            res[find(first_query)].add(second_query)

    with open(save_path, "a", encoding="utf-8") as save_file:
        print("正在处理正样本")
        for key, value in res.items():
            elements = list(value)
            length = len(elements)
            for i in range(length):
                for j in range(i + 1, length):
                    save_file.write(elements[i] + "\t" + elements[j] + "\t1" + "\n")
                    save_file.write(elements[j] + "\t" + elements[i] + "\t1" + "\n")
                    count += 2
            if if_self:
                for element in elements:
                    save_file.write(element + "\t" + element + "\t1" + "\n")
                    count += 1

            if count % 1000 == 0:
                print("\r已处理 {} 条query-pairs".format(count), end="", flush=True)

        print("\n正在处理负样本")
        count = 0
        for first, second in negative:
            save_file.write(first + "\t" + second + "\t0" + "\n")
            save_file.write(second + "\t" + first + "\t0" + "\n")

            count += 2
            if count % 1000 == 0:
                print("\r已处理 {} 条query-pairs".format(count), end="", flush=True)

        if if_self:
            print("\n正在处理负样本转化正样本")
            count = 0
            for ne_element in negative_set:
                save_file.write(ne_element + "\t" + ne_element + "\t1" + "\n")

                count += 1
                if count % 1000 == 0:
                    print("\r已处理 {} 条query-pairs".format(count), end="", flush=True)
