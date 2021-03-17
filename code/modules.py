from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import zipfile
import numpy as np
import tensorflow as tf
from code.optimizers import CustomSchedule
from code.tools import get_dict_string
from code.tools import ProgressBar
from typing import *


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
        for (batch, (train_enc, train_dec, month_enc, month_dec, labels)) in enumerate(
                train_dataset.take(max_train_steps)):
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


def evaluate(model: tf.keras.Model, batch_size: Any, dataset: Any, *args, **kwargs) -> Dict:
    """ 评估器

    :param model: 评估模型
    :param batch_size: batch大小
    :param dataset: 验证数据集
    :return: 评估指标
    """
    progress_bar = ProgressBar()
    loss_metric = tf.keras.metrics.Mean(name="evaluate_loss")
    steps_per_epoch = 188 // batch_size
    progress_bar.reset(total=steps_per_epoch, num=batch_size)

    valid_metrics = _valid_step(model=model, dataset=dataset,
                                progress_bar=progress_bar, loss_metric=loss_metric, **kwargs)

    return valid_metrics


def make_zip(source_dir, output_filename):
    zip_file = zipfile.ZipFile(output_filename, "w")
    pre_len = len(os.path.dirname(source_dir))
    for parent, dir_names, filenames in os.walk(source_dir):
        for filename in filenames:
            path_file = os.path.join(parent, filename)
            arc_name = path_file[pre_len:].strip(os.path.sep)  # 相对路径
            zip_file.write(path_file, arc_name)
    zip_file.close()


def inference(model: tf.keras.Model, result_save_path: AnyStr, test_data_path: AnyStr = None) -> NoReturn:
    """ 推断器

    :param model: 推断模型
    :param result_save_path: 推断结果保存路径
    :param test_data_path: 推断用TFRecord文件路径
    :return:
    """
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)

    all_file = os.listdir(test_data_path)
    file_list = []
    for name in all_file:
        if os.path.splitext(name)[1] == ".npy":
            file_list.append(name)

    for test_file in file_list:
        start = int(os.path.splitext(name)[0].split("-")[1])

        train_enc = np.load(file=test_data_path + test_file)
        train_dec = np.concatenate([train_enc[6:, :, :, :], np.zeros(shape=(24, 24, 72, 4), dtype=np.float)])
        month_enc = np.array([[(month % 12) + 1 for month in range(start - 1, start + 11)]])
        month_dec = np.array([[(month % 12) + 1 for month in range(start + 5, start + 35)]])

        outputs = model(inputs=[train_enc, train_dec, month_enc, month_dec])
        treat_outputs = outputs[0, -24:, 0]

        np.save(file=result_save_path + test_file, arr=treat_outputs.numpy())

    make_zip(source_dir=result_save_path, output_filename="result.zip")
