from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
from typing import *
import tensorflow as tf
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import TFBertForMaskedLM
from transformers import TFBertModel


def load_data(filename: AnyStr, data_type: AnyStr):
    """加载数据

    :param filename: 文件名
    :param data_type: 数据类型
    """

    count = 0
    queries_labels = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().split('\t')
            if len(line) == 3:
                first_query, second_query, label = line[0], line[1], int(line[2])
            else:
                first_query, second_query, label = line[0], line[1], -5
            first_query = [int(i) for i in first_query.split(' ')]
            second_query = [int(i) for i in second_query.split(' ')]
            queries_labels.append((first_query, second_query, label))
            count += 1
            if count % 100 == 0:
                print("\r已读取 {} 条{}query-pairs".format(count, data_type), end="", flush=True)
    return queries_labels


def load_vocab(dict_path):
    token_dict = {}
    with open(dict_path, "r", encoding="utf-8") as dict_file:
        for line in dict_file:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)

    return token_dict


def preprocess(train_data_path: AnyStr, test_data_path: AnyStr):
    data = load_data(train_data_path, "训练")
    print("\n训练数据读取完毕")
    train_data = [d for i, d in enumerate(data) if i % 10 != 0]
    valid_data = [d for i, d in enumerate(data) if i % 10 == 0]
    test_data = load_data(test_data_path, "验证")
    print("\n验证数据读取完毕")

    # 模拟未标注
    for queries in valid_data + test_data:
        train_data.append((queries[0], queries[1], -5))

    return data, train_data, valid_data, test_data


def statistics(min_freq: Any, dict_path: AnyStr, data: Any):
    print("正在统计词频")
    tokens, count = {}, 0
    for first_query, second_query, label in data:
        for i in first_query + second_query:
            tokens[i] = tokens.get(i, 0) + 1
        count += 1
        if count % 100 == 0:
            print("\r已统计 {} 条query-pairs".format(count), end="", flush=True)

    tokens = {i: j for i, j in tokens.items() if j >= min_freq}
    tokens = sorted(tokens.items(), key=lambda s: -s[1])
    tokens = {t[0]: i + 7 for i, t in enumerate(tokens)}  # 0: pad, 1: unk, 2: cls, 3: sep, 4: mask, 5: no, 6: yes

    # BERT词频
    counts = json.load(open("../tcdata/counts.json", "r", encoding="utf-8"))
    del counts["[CLS]"]
    del counts["[SEP]"]
    token_dict = load_vocab(dict_path)
    freq = [counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])]
    keep_tokens = list(np.argsort(freq)[::-1])

    return tokens, keep_tokens


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
    data_v, train_data_v, valid_data_v, test_data_v = preprocess(
        train_data_path="../tcdata/gaiic_track3_round1_train_20210228.tsv",
        test_data_path="../tcdata/gaiic_track3_round1_testA_20210228.tsv")
    tokens, keep_tokens = statistics(min_freq=5, dict_path="../tcdata/bert/vocab.txt", data=data_v + test_data_v)

    for index, (first_query, second_query, label) in enumerate(data_v):
        print(first_query, second_query, label)
        va

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


if __name__ == '__main__':
    model_path = "../tcdata/bert/"
    tokenizer = BertTokenizer.from_pretrained("../tcdata/bert/vocab.txt")
    model_config = BertConfig.from_pretrained("../tcdata/bert/config.json")
    # model_config.output_attentions = False
    # model_config.output_hidden_states = False
    # model_config.use_cache = True
    # #
    # bert_model = TFBertModel.from_pretrained(pretrained_model_name_or_path=model_path, from_pt=False,
    #                                          config=model_config, cache_dir="../user_data/temp")
    # model = TFBertForMaskedLM(config=model_config)
    # model.bert = bert_model
    # model.resize_token_embeddings(len(tokenizer))
    model = TFBertForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_path, from_pt=False,
                                              config=model_config, cache_dir="../user_data/temp")
    model.resize_token_embeddings(len(tokenizer))
    #

    # inputs = tokenizer("中国的首都是[MASK]", return_tensors="tf")
    # inputs["labels"] = tokenizer("中国的首都是北京", return_tensors="tf")["input_ids"]
    inputs = tokenizer.encode("中国的首都是[MASK]", return_tensors="tf")
    # print(tokenizer.tokenize("中国的首都是[MASK]"))
    outputs = model(inputs)
    # print(outputs)
    # exit(0)
    o1 = tf.argmax(outputs.logits[0], axis=1)
    print(o1)
    print(tokenizer.decode(o1))


    #
    # model()

    # text = "今天天气真好，我们一起出去玩吧"
    # text1 = "你是谁"
    # input_ids = tokenizer.encode("[unused5]")
    # tokens = tokenizer.tokenize(text)
    # tokens_id = tokenizer.convert_tokens_to_ids(tokens)
    # print(tokenizer.build_inputs_with_special_tokens(tokens_id))
    # print(tokenizer.encode(text))
    # print(model.predict(input_ids))
    # print(input_ids)

    # train_dataset = load_raw_dataset(data_path=train_data_path, max_sentence_length=max_sentence_length,
    #                                  batch_size=batch_size, buffer_size=buffer_size, data_type="train")
