from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from code.optimizers import CustomSchedule
from code.tools import get_dict_string
from code.tools import ProgressBar
from sklearn.metrics import roc_auc_score
from typing import *
from transformers import BertConfig
from transformers import TFBertModel
from transformers import create_optimizer


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
    counts = json.load(open("./tcdata/counts.json", "r", encoding="utf-8"))
    del counts["[CLS]"]
    del counts["[SEP]"]
    token_dict = load_vocab(dict_path)
    freq = [counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])]
    keep_tokens = list(np.argsort(freq)[::-1])

    dict_keep_tokens = {t: i for i, t, in enumerate([0, 100, 101, 102, 103, 100, 100] + keep_tokens[:len(tokens)])}

    return tokens, [0, 100, 101, 102, 103, 100, 100] + keep_tokens[:len(tokens)], dict_keep_tokens


def random_mask(text_ids, length, keep_tokens, dict_keep_tokens):
    """随机mask
    """
    input_ids, output_ids = [], []
    rands = np.random.random(len(text_ids))
    for r, i in zip(rands, text_ids):
        if r < 0.15 * 0.8:
            input_ids.append(103)
            output_ids.append(dict_keep_tokens.get(i, 1))
        elif r < 0.15 * 0.9:
            input_ids.append(i)
            output_ids.append(dict_keep_tokens.get(i, 1))
        elif r < 0.15:
            input_ids.append(keep_tokens[np.random.choice(length) + 7])
            output_ids.append(dict_keep_tokens.get(i, 1))
        else:
            input_ids.append(i)
            output_ids.append(0)
    return input_ids, output_ids


def sample_convert(first_query, second_query, label, tokens, keep_tokens, dict_keep_tokens):
    """转换为MLM格式
    """
    length = len(tokens)
    text1_ids = [keep_tokens[tokens.get(t, 1)] for t in first_query]
    text2_ids = [keep_tokens[tokens.get(t, 1)] for t in second_query]

    if np.random.random() < 0.5:
        text1_ids, text2_ids = text2_ids, text1_ids

    text1_ids, first_output = random_mask(text1_ids, 6925, keep_tokens, dict_keep_tokens)
    text2_ids, second_output = random_mask(text2_ids, 6925, keep_tokens, dict_keep_tokens)

    queries = [101] + text1_ids + [102] + text2_ids + [102]
    segments = [0] * len(queries)
    outputs = [label + 5] + first_output + [0] + second_output + [0]
    return queries, segments, outputs


def load_raw_dataset(train_data_path: AnyStr, max_sentence_length: Any, batch_size: Any, buffer_size: Any,
                     dict_path: AnyStr, test_data_path: AnyStr, reshuffle_each_iteration: Any = True) -> Tuple:
    """ 处理原始数据，并将处理后的数据保存为TFRecord格式，两个句子合并

    :param train_data_path: 原始数据路径
    :param max_sentence_length: 最大输入数据
    :param batch_size: batch大小
    :param buffer_size: 缓冲大小
    :param dict_path: 词表文件
    :param test_data_path: 原始验证数据
    :param reshuffle_each_iteration: 是否每个epoch打乱
    :return: 无返回值
    """
    data_v, train_data_v, valid_data_v, test_data_v = preprocess(train_data_path=train_data_path,
                                                                 test_data_path=test_data_path)
    tokens, keep_tokens, dict_keep_tokens = statistics(min_freq=5, dict_path=dict_path, data=data_v + test_data_v)

    # def generator():
    #     for index, (first_query, second_query, label) in enumerate(data_v):
    #         queries, segments, labels = sample_convert(first_query, second_query, label, tokens, keep_tokens)
    #
    #         queries = tf.keras.preprocessing.sequence.pad_sequences([queries], max_sentence_length, padding="post")
    #         segments = tf.keras.preprocessing.sequence.pad_sequences([segments], max_sentence_length, padding="post")
    #         labels = tf.keras.preprocessing.sequence.pad_sequences([labels], max_sentence_length, padding="post")
    #
    #         yield tf.squeeze(queries, axis=0), tf.squeeze(segments, axis=0), tf.squeeze(labels, axis=0), label
    train_queries = tf.constant(value=0, shape=(1, max_sentence_length), dtype=tf.int32)
    train_segments = tf.constant(value=0, shape=(1, max_sentence_length), dtype=tf.int32)
    train_labels = tf.constant(value=0, shape=(1, max_sentence_length), dtype=tf.int32)
    count = 0
    print("正在加载训练数据")
    for index, (first_query, second_query, label) in enumerate(train_data_v):
        queries, segments, labels = sample_convert(first_query, second_query, label, tokens, keep_tokens, dict_keep_tokens)

        queries = tf.keras.preprocessing.sequence.pad_sequences([queries], max_sentence_length, padding="post")
        segments = tf.keras.preprocessing.sequence.pad_sequences([segments], max_sentence_length, padding="post")
        labels = tf.keras.preprocessing.sequence.pad_sequences([labels], max_sentence_length, padding="post")

        train_queries = tf.concat([train_queries, queries], axis=0)
        train_segments = tf.concat([train_segments, segments], axis=0)
        train_labels = tf.concat([train_labels, labels], axis=0)

        count += 1
        if count % 100 == 0:
            print("\r已处理 {} 条训练数据".format(count), end="", flush=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_queries[1:], train_segments[1:], train_labels[1:]))

    train_dataset = train_dataset.shuffle(
        buffer_size=buffer_size, reshuffle_each_iteration=reshuffle_each_iteration
    ).prefetch(tf.data.experimental.AUTOTUNE).batch(batch_size, drop_remainder=True)

    valid_queries = tf.constant(value=0, shape=(1, max_sentence_length), dtype=tf.int32)
    valid_segments = tf.constant(value=0, shape=(1, max_sentence_length), dtype=tf.int32)
    valid_labels = []
    count = 0
    print("正在加载验证数据")
    for index, (first_query, second_query, label) in enumerate(valid_data_v):
        text1_ids = [keep_tokens[tokens.get(t, 1)] for t in first_query]
        text2_ids = [keep_tokens[tokens.get(t, 1)] for t in second_query]

        queries = [101] + text1_ids + [102] + text2_ids + [102]
        segments = [0] * len(queries)

        queries = tf.keras.preprocessing.sequence.pad_sequences([queries], max_sentence_length, padding="post")
        segments = tf.keras.preprocessing.sequence.pad_sequences([segments], max_sentence_length, padding="post")

        valid_queries = tf.concat([valid_queries, queries], axis=0)
        valid_segments = tf.concat([valid_segments, segments], axis=0)
        valid_labels.append(label)

        count += 1
        if count % 100 == 0:
            print("\r已处理 {} 条验证数据".format(count), end="", flush=True)

    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_queries[1:], valid_segments[1:], valid_labels))

    valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True)

    return train_dataset, valid_dataset


def bert_model(vocab_size: Any, bert: Any) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)
    segments = tf.keras.Input(shape=(None,), dtype=tf.int32)

    # bert_outputs = bert(input_ids=inputs, token_type_ids=segments)
    bert_outputs = bert(inputs)[0]
    # outputs = tf.keras.layers.Dense(units=vocab_size, activation="relu")(bert_outputs.last_hidden_state)
    outputs = tf.keras.layers.Dense(units=vocab_size, activation="relu")(bert_outputs)

    return tf.keras.Model(inputs=[inputs, segments], outputs=outputs)


def train(model: tf.keras.Model, checkpoint: tf.train.CheckpointManager, batch_size: Any, buffer_size: Any,
          epochs: Any, train_data_path: AnyStr, test_data_path: AnyStr, dict_path: AnyStr, max_sentence_length: Any,
          max_train_steps: Any = -1, checkpoint_save_freq: Any = 2, *args, **kwargs) -> Dict:
    """ 训练器

    :param model: 训练模型
    :param checkpoint: 检查点管理器
    :param batch_size: batch 大小
    :param buffer_size: 缓冲大小
    :param epochs: 训练周期
    :param train_data_path: 训练数据保存路径
    :param test_data_path: 测试数据保存路径
    :param dict_path: 词表文件
    :param max_sentence_length: 最大句子对长度
    :param max_train_steps: 最大训练数据量，-1为全部
    :param checkpoint_save_freq: 检查点保存频率
    :return:
    """
    print("训练开始，正在准备数据中")

    loss_metric = tf.keras.metrics.Mean(name="train_loss_metric")

    optimizer = tf.optimizers.Adam(learning_rate=1e-5)

    train_steps_per_epoch = 125000 // batch_size
    valid_steps_per_epoch = 10000 // batch_size
    warmup_steps = train_steps_per_epoch // 3
    total_steps = train_steps_per_epoch * epochs - warmup_steps

    # learning_rate = CustomSchedule(d_model=768, warmup_steps=warmup_steps)
    # optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, name="optimizer")

    # optimizer, _ = create_optimizer(init_lr=2e-5, num_train_steps=total_steps, num_warmup_steps=warmup_steps)

    train_dataset, valid_dataset = load_raw_dataset(
        train_data_path=train_data_path, max_sentence_length=max_sentence_length, batch_size=batch_size,
        buffer_size=buffer_size, dict_path=dict_path, test_data_path=test_data_path)

    print("训练开始")
    progress_bar = ProgressBar()
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        start_time = time.time()
        loss_metric.reset_states()
        progress_bar.reset(total=train_steps_per_epoch, num=batch_size)

        train_metric = None
        for (batch, (queries, segments, labels)) in enumerate(train_dataset.take(max_train_steps)):
            train_metric = _train_step(
                model=model, optimizer=optimizer, loss_metric=loss_metric,
                queries=queries, segments=segments, targets=labels
            )
            progress_bar(current=batch + 1, metrics=get_dict_string(data=train_metric))

        progress_bar(current=progress_bar.total, metrics=get_dict_string(data=train_metric))

        progress_bar.done(step_time=time.time() - start_time)

        if (epoch + 1) % checkpoint_save_freq == 0:
            checkpoint.save()

            if valid_steps_per_epoch == 0 or valid_dataset is None:
                print("验证数据量过小，小于batch_size，已跳过验证轮次")
            else:
                progress_bar.reset(total=valid_steps_per_epoch, num=batch_size)
                valid_metrics = _valid_step(model=model, dataset=valid_dataset,
                                            progress_bar=progress_bar, loss_metric=loss_metric, **kwargs)
    print("训练结束")
    return {}


def loss_function(pred: Any, targets: Any):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(targets, pred)
    mask = tf.cast(x=tf.math.not_equal(targets, 0), dtype=tf.float32)
    total = tf.reduce_sum(mask)

    return tf.divide(tf.reduce_sum(mask * loss), total)


@tf.function(autograph=True)
def _train_step(model: tf.keras.Model, optimizer: Any, segments: Any,
                loss_metric: tf.keras.metrics.Mean, queries: Any, targets: Any) -> Any:
    """ 单个训练步

    :param model: 训练模型
    :param optimizer: 优化器
    :param loss_metric: 损失计算器
    :param queries: 第一个查询句子
    :param targets: 第二个查询句子
    :return: 训练指标
    """
    with tf.GradientTape() as tape:
        outputs = model(inputs=[queries, segments])
        loss = loss_function(pred=outputs, targets=targets)
    loss_metric(loss)
    variables = model.trainable_variables
    gradients = tape.gradient(target=loss, sources=variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return {"train_loss": loss_metric.result()}


def _valid_step(model: tf.keras.Model, dataset: tf.data.Dataset, progress_bar: ProgressBar,
                loss_metric: tf.keras.metrics.Mean, max_train_steps: Any = -1) -> Dict:
    """ 验证步

    :param model: 验证模型
    :param dataset: 验证数据集
    :param progress_bar: 进度管理器
    :param batch_size: batch大小
    :param loss_metric: 损失计算器
    :param max_train_steps: 验证步数
    :return: 验证指标
    """
    print("验证轮次")
    start_time = time.time()
    loss_metric.reset_states()
    result, targets = tf.convert_to_tensor([], dtype=tf.float32), tf.convert_to_tensor([], dtype=tf.int32)
    for (batch, (queries, segments, labels)) in enumerate(dataset.take(max_train_steps)):
        outputs = model(inputs=[queries, segments])

        loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(labels,
                                                                                                       outputs[:, 0, :])
        loss_metric(tf.divide(tf.reduce_sum(loss), labels.shape[0]))

        result = tf.concat([result, tf.nn.softmax(logits=outputs[:, 0, 5:7], axis=-1)[:, 1]], axis=0)
        targets = tf.concat([targets, labels], axis=0)

        progress_bar(current=batch + 1, metrics=get_dict_string(data={"valid_loss": loss_metric.result()}))

    auc_score = roc_auc_score(y_true=targets, y_score=result)
    progress_bar(current=progress_bar.total, metrics=get_dict_string(
        data={"valid_loss": loss_metric.result(), "valid_auc": auc_score}
    ))

    progress_bar.done(step_time=time.time() - start_time)

    return {"valid_loss": loss_metric.result(), "valid_auc": auc_score}


def evaluate(model: tf.keras.Model, batch_size: Any, buffer_size: Any, max_sentence_length: Any,
             train_data_path: Any, *args, **kwargs) -> Dict:
    """ 评估器

    :param model: 评估模型
    :param batch_size: batch大小
    :param buffer_size: 缓冲大小
    :param train_data_path: TFRecord数据文件路径
    :return: 评估指标
    """
    progress_bar = ProgressBar()
    loss_metric = tf.keras.metrics.Mean(name="evaluate_loss")

    dataset = load_raw_dataset(data_path=train_data_path, max_sentence_length=max_sentence_length,
                               batch_size=batch_size, buffer_size=buffer_size, data_type="valid")
    steps_per_epoch = 10000 // batch_size
    progress_bar.reset(total=steps_per_epoch, num=batch_size)

    valid_metrics = _valid_step(model=model, dataset=dataset, batch_size=batch_size,
                                progress_bar=progress_bar, loss_metric=loss_metric, **kwargs)

    return valid_metrics


def inference(model: tf.keras.Model, batch_size: Any, buffer_size: Any, result_save_path: AnyStr,
              record_data_path: AnyStr = None, raw_data_path: AnyStr = None,
              max_sentence: int = 30, max_step: Any = 10) -> NoReturn:
    """ 推断器

    :param model: 推断模型
    :param batch_size: batch大小
    :param buffer_size: 缓冲大小
    :param result_save_path: 推断结果保存路径
    :param record_data_path: 推断用TFRecord文件路径
    :param raw_data_path: 原始数据集路径，预留赛事指定推断方式
    :param dict_path: 字典路径
    :param max_sentence: 最大句子长度
    :param max_step: 最大时间步
    :return:
    """
    count = 0
    if record_data_path is not None:
        dataset = load_dataset(record_path=record_data_path, batch_size=batch_size,
                               buffer_size=buffer_size, data_type="valid", drop_remainder=False)
        with open(result_save_path, "a", encoding="utf-8") as file:
            for batch, (first_queries, second_queries, _) in enumerate(dataset.take(max_step)):
                result = inference_step(model=model, first_queries=first_queries, second_queries=second_queries)
                for i in range(batch_size):
                    file.write("{:.3f}\n".format(result[i, 1]))

                count += batch_size
                print("\r已推断 {} 条query-pairs".format(count), end="", flush=True)
    elif raw_data_path is not None:
        # tokenizer = load_tokenizer(dict_path=dict_path)
        with open(raw_data_path, "r", encoding="utf-8") as file, open(result_save_path, "a",
                                                                      encoding="utf-8") as result_file:
            for line in file:
                line = line.strip().strip("\n").split("\t")
                if len(line) != 2:
                    raise ValueError("推断数据集出现残缺")

                # first_query = " ".join(jieba.cut(line[0]))
                # second_query = " ".join(jieba.cut(line[1]))
                # first_query = tokenizer.texts_to_sequences([first_query])
                # second_query = tokenizer.texts_to_sequences([second_query])
                first_query = [line[0].split(" ")]
                second_query = [line[1].split(" ")]

                first_query = pad_sequences(first_query, maxlen=max_sentence, padding="post")
                second_query = pad_sequences(second_query, maxlen=max_sentence, padding="post")

                result = inference_step(model=model, first_queries=first_query, second_queries=second_query)
                result_file.write("{:.3f}\n".format(result[0, 1]))

                count += 1
                if count % 100 == 0:
                    print("\r已推断 {} 条query-pairs".format(count), end="", flush=True)
    else:
        raise ValueError("推断数据集路径出现问题，请检查后重试")


@tf.function(autograph=True)
def inference_step(model: tf.keras.Model, first_queries: tf.Tensor, second_queries: tf.Tensor) -> Any:
    """ 单个推断步

    :param model: 推断模型
    :param first_queries: 第一个查询语句
    :param second_queries: 第二个查询语句
    :return: 推断结果
    """
    outputs = model(inputs=[first_queries, second_queries])
    return outputs
