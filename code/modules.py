from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import jieba
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from code.optimizers import CustomSchedule
from code.tools import get_dict_string
from code.preprocess import preprocess_data_diff
from code.tools import ProgressBar
from sklearn.metrics import roc_auc_score
from typing import Any
from typing import AnyStr
from typing import Dict
from typing import NoReturn
from typing import Tuple


def train(model: tf.keras.Model, checkpoint: tf.train.CheckpointManager, batch_size: Any,
          epochs: Any, train_dataset: Any, valid_dataset: AnyStr, max_train_steps: Any = -1,
          checkpoint_save_freq: Any = 2, *args, **kwargs) -> Dict:
    """ 训练器

    :param model: 训练模型
    :param checkpoint: 检查点管理器
    :param batch_size: batch 大小
    :param epochs: 训练周期
    :param train_dataset: 训练数据集
    :param valid_dataset: 验证数据集
    :param max_train_steps: 最大训练数据量，-1为全部
    :param checkpoint_save_freq: 检查点保存频率
    :return:
    """
    print("训练开始，正在准备数据中")
    # learning_rate = CustomSchedule(d_model=embedding_dim)
    loss_metric = tf.keras.metrics.Mean(name="train_loss_metric")
    optimizer = tf.optimizers.Adam(learning_rate=2e-5, beta_1=0.9, beta_2=0.999, name="optimizer")

    train_steps_per_epoch = max_train_steps if max_train_steps != -1 else (1000 // batch_size)
    valid_steps_per_epoch = 188 // batch_size

    progress_bar = ProgressBar()
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        start_time = time.time()
        loss_metric.reset_states()
        progress_bar.reset(total=train_steps_per_epoch, num=batch_size)

        train_metric = None
        for (batch, (train_enc, train_dec, month_enc, month_dec, labels)) in enumerate(train_dataset.take(max_train_steps)):
            train_metric, prediction = _train_step(
                model=model, optimizer=optimizer, loss_metric=loss_metric, train_enc=train_enc,
                train_dec=train_dec, month_enc=month_enc, month_dec=month_dec, labels=labels
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


@tf.function(autograph=True)
def _train_step(model: tf.keras.Model, optimizer: tf.keras.optimizers.Adam, loss_metric: tf.keras.metrics.Mean,
                train_enc: Any, train_dec: Any, month_enc: Any, month_dec: Any, labels: Any) -> Tuple:
    """ 单个训练步

    :param model: 训练模型
    :param optimizer: 优化器
    :param loss_metric: 损失计算器
    :param train_enc: enc输入
    :param train_dec: dec输入
    :param month_enc: enc月份输入
    :param month_dec: dec月份输入
    :param labels: 标签
    :return: 训练指标
    """
    with tf.GradientTape() as tape:
        train_enc = tf.squeeze(train_enc, axis=0)
        train_dec = tf.squeeze(train_dec, axis=0)
        outputs = model(inputs=[train_enc, train_dec, month_enc, month_dec])
        treat_outputs = tf.squeeze(input=outputs[:, -24:, :], axis=-1)
        loss = tf.keras.losses.MSE(labels, treat_outputs)
    loss_metric(loss)
    variables = model.trainable_variables
    gradients = tape.gradient(target=loss, sources=variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return {"train_loss": loss_metric.result()}, treat_outputs


def _valid_step(model: tf.keras.Model, dataset: tf.data.Dataset, progress_bar: ProgressBar,
                loss_metric: tf.keras.metrics.Mean, max_train_steps: Any = -1) -> Dict:
    """ 验证步

    :param model: 验证模型
    :param dataset: 验证数据集
    :param progress_bar: 进度管理器
    :param loss_metric: 损失计算器
    :param max_train_steps: 验证步数
    :return: 验证指标
    """
    print("验证轮次")
    start_time = time.time()
    loss_metric.reset_states()

    for (batch, (train_enc, train_dec, month_enc, month_dec, labels)) in enumerate(dataset.take(max_train_steps)):
        train_enc = tf.squeeze(train_enc, axis=0)
        train_dec = tf.squeeze(train_dec, axis=0)
        outputs = model(inputs=[train_enc, train_dec, month_enc, month_dec])
        treat_outputs = tf.squeeze(input=outputs[:, -24:, :], axis=-1)
        loss = tf.keras.losses.MSE(labels, treat_outputs)

        loss_metric(loss)

        progress_bar(current=batch + 1, metrics=get_dict_string(data={"valid_loss": loss_metric.result()}))

    progress_bar(current=progress_bar.total, metrics=get_dict_string(
        data={"valid_loss": loss_metric.result()}
    ))

    progress_bar.done(step_time=time.time() - start_time)

    return {"valid_loss": loss_metric.result()}


def evaluate(model: tf.keras.Model, batch_size: Any, buffer_size: Any, record_data_path: Any, *args, **kwargs) -> Dict:
    """ 评估器

    :param model: 评估模型
    :param batch_size: batch大小
    :param buffer_size: 缓冲大小
    :param record_data_path: TFRecord数据文件路径
    :return: 评估指标
    """
    progress_bar = ProgressBar()
    loss_metric = tf.keras.metrics.Mean(name="evaluate_loss")

    dataset = load_dataset(record_path=record_data_path, batch_size=batch_size,
                           buffer_size=buffer_size, data_type="valid")
    steps_per_epoch = 10000 // batch_size
    progress_bar.reset(total=steps_per_epoch, num=batch_size)

    valid_metrics = _valid_step(model=model, dataset=dataset,
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


def inference1(model: tf.keras.Model, batch_size: Any, buffer_size: Any, result_save_path: AnyStr,
               record_data_path: AnyStr = None, raw_data_path: AnyStr = None,
               max_sentence: int = 60, max_step: Any = 10) -> NoReturn:
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
                first = line[0].split(" ")
                second = line[1].split(" ")

                first_query = [[21963] + first + [21964] + second + [21964]]
                second_query = [[1 for _ in range(len(first) + 2)] + [2 for _ in range(len(second) + 1)]]

                first_query = pad_sequences(first_query, maxlen=max_sentence, padding="post")
                second_query = pad_sequences(second_query, maxlen=max_sentence, padding="post")

                result = inference_step(model=model, first_queries=first_query, second_queries=second_query)
                result_file.write("{:.3f}\n".format(result[0, 1]))

                count += 1
                if count % 100 == 0:
                    print("\r已推断 {} 条query-pairs".format(count), end="", flush=True)
    else:
        raise ValueError("推断数据集路径出现问题，请检查后重试")
