from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from code.optimizers import CustomSchedule
from code.tools import get_dict_string
from code.tools import load_pair_dataset
from code.tools import ProgressBar
from sklearn.metrics import roc_auc_score
from typing import Any
from typing import AnyStr
from typing import Dict
from typing import NoReturn
from typing import Tuple


def train(model: tf.keras.Model, checkpoint: tf.train.CheckpointManager, batch_size: Any, buffer_size: Any,
          epochs: Any, embedding_dim: Any, train_data_path: AnyStr, valid_data_path: AnyStr,
          max_sentence_length=Any, max_train_steps: Any = -1, checkpoint_save_freq: Any = 2, *args, **kwargs) -> Dict:
    """ 训练器

    :param model: 训练模型
    :param checkpoint: 检查点管理器
    :param batch_size: batch 大小
    :param buffer_size: 缓冲大小
    :param epochs: 训练周期
    :param embedding_dim: 词嵌入大小
    :param train_data_path: 训练数据文件保存路径
    :param valid_data_path: 验证数据文件保存路径
    :param max_sentence_length: 最大句子长度
    :param max_train_steps: 最大训练数据量，-1为全部
    :param checkpoint_save_freq: 检查点保存频率
    :return:
    """
    print("训练开始，正在准备数据中")
    # learning_rate = CustomSchedule(d_model=embedding_dim)
    loss_metric = tf.keras.metrics.Mean(name="train_loss_metric")
    optimizer = tf.optimizers.Adam(learning_rate=2e-5, beta_1=0.9, beta_2=0.98, name="optimizer")

    train_dataset = load_pair_dataset(data_path=train_data_path, max_sentence_length=max_sentence_length,
                                      batch_size=batch_size, buffer_size=buffer_size, data_type="train")
    valid_dataset = load_pair_dataset(data_path=valid_data_path, max_sentence_length=max_sentence_length,
                                      batch_size=batch_size, buffer_size=buffer_size, data_type="valid")
    train_steps_per_epoch = max_train_steps if max_train_steps != -1 else (206588 // batch_size)
    valid_steps_per_epoch = 10000 // batch_size

    progress_bar = ProgressBar()
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        start_time = time.time()
        loss_metric.reset_states()
        progress_bar.reset(total=train_steps_per_epoch, num=batch_size)

        train_metric = None
        for (batch, (first_queries, second_queries, labels)) in enumerate(train_dataset.take(max_train_steps)):
            train_metric, prediction = _train_step(
                model=model, optimizer=optimizer, loss_metric=loss_metric,
                first_queries=first_queries, second_queries=second_queries, labels=labels
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
def _train_step(model: tf.keras.Model, optimizer: tf.keras.optimizers.Adam,
                loss_metric: tf.keras.metrics.Mean, first_queries: Any, second_queries: Any, labels: Any) -> Tuple:
    """ 单个训练步

    :param model: 训练模型
    :param optimizer: 优化器
    :param loss_metric: 损失计算器
    :param first_queries: 第一个查询句子
    :param second_queries: 第二个查询句子
    :param labels: 标签
    :return: 训练指标
    """
    with tf.GradientTape() as tape:
        outputs = model(inputs=[first_queries, second_queries])
        loss = tf.keras.losses.SparseCategoricalCrossentropy()(labels, outputs)
    loss_metric(loss)
    variables = model.trainable_variables
    gradients = tape.gradient(target=loss, sources=variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return {"train_loss": loss_metric.result()}, outputs


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
    result, targets = tf.convert_to_tensor([], dtype=tf.float32), tf.convert_to_tensor([], dtype=tf.int32)

    for (batch, (first_queries, second_queries, labels)) in enumerate(dataset.take(max_train_steps)):
        outputs = model(inputs=[first_queries, second_queries])

        loss = tf.keras.losses.SparseCategoricalCrossentropy()(labels, outputs)
        loss_metric(loss)

        result = tf.concat([result, outputs[:, 1]], axis=0)
        targets = tf.concat([targets, labels], axis=0)

        progress_bar(current=batch + 1, metrics=get_dict_string(data={"valid_loss": loss_metric.result()}))

    auc_score = roc_auc_score(y_true=targets, y_score=result)
    progress_bar(current=progress_bar.total, metrics=get_dict_string(
        data={"valid_loss": loss_metric.result(), "valid_auc": auc_score}
    ))

    progress_bar.done(step_time=time.time() - start_time)

    return {"valid_loss": loss_metric.result(), "valid_auc": auc_score}


def evaluate(model: tf.keras.Model, batch_size: Any, buffer_size: Any,
             valid_data_path: Any, max_sentence_length: Any, *args, **kwargs) -> Dict:
    """ 评估器

    :param model: 评估模型
    :param batch_size: batch大小
    :param buffer_size: 缓冲大小
    :param valid_data_path: 评估数据文件路径
    :param max_sentence_length: 句子最大长度
    :return: 评估指标
    """
    progress_bar = ProgressBar()
    loss_metric = tf.keras.metrics.Mean(name="evaluate_loss")

    valid_dataset = load_pair_dataset(data_path=valid_data_path, max_sentence_length=max_sentence_length,
                                      batch_size=batch_size, buffer_size=buffer_size, data_type="valid")
    steps_per_epoch = 10000 // batch_size
    progress_bar.reset(total=steps_per_epoch, num=batch_size)

    valid_metrics = _valid_step(model=model, dataset=valid_dataset,
                                progress_bar=progress_bar, loss_metric=loss_metric, **kwargs)

    return valid_metrics


def inference(model: tf.keras.Model, result_save_path: AnyStr,
              raw_data_path: AnyStr = None, max_sentence: int = 30) -> NoReturn:
    """ 推断器

    :param model: 推断模型
    :param result_save_path: 推断结果保存路径
    :param raw_data_path: 原始数据集路径，预留赛事指定推断方式
    :param max_sentence: 最大句子长度
    :return:
    """
    count = 0
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError("测试文本不存在，请检查后重试")

    with open(raw_data_path, "r", encoding="utf-8") as file, open(result_save_path, "a",
                                                                  encoding="utf-8") as result_file:
        for line in file:
            line = line.strip().strip("\n").split("\t")
            if len(line) != 2:
                raise ValueError("推断数据集出现残缺")

            first_query = [[int(num) for num in line[0].split(" ")]]
            second_query = [[int(num) for num in line[1].split(" ")]]

            first_query = pad_sequences(first_query, maxlen=max_sentence, padding="post")
            second_query = pad_sequences(second_query, maxlen=max_sentence, padding="post")

            result = inference_step(model=model, first_queries=first_query, second_queries=second_query)
            result_file.write("{:.3f}\n".format(result[0, 1]))

            count += 1
            if count % 100 == 0:
                print("\r已推断 {} 条query-pairs".format(count), end="", flush=True)


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
