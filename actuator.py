# Copyright 2021 DengBoCong. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Informer的实现执行器入口
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from model import informer
from tools import make_zip
from tools import get_dict_string
from tools import load_checkpoint
from tools import ProgressBar
from tools import eval_score
from preprocess import preprocess_soda
from preprocess import preprocess_cmip
from preprocess import load_dataset
from argparse import ArgumentParser
from typing import Any
from typing import Dict
from typing import Tuple
from typing import AnyStr
from typing import NoReturn


def train(model: tf.keras.Model, checkpoint: tf.train.CheckpointManager, batch_size: Any,
          epochs: Any, train_dataset: Any, valid_dataset: AnyStr = None, max_train_steps: Any = -1,
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

    train_steps_per_epoch = max_train_steps if max_train_steps != -1 else (40000 // batch_size)
    valid_steps_per_epoch = 3944 // batch_size

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


# @tf.function(autograph=True)
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
        loss = tf.keras.losses.mean_squared_error(labels, treat_outputs)

        # acikill_metrics = eval_score(preds=outputs, label=labels)
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
        start = int(os.path.splitext(name)[0].split("_")[2])

        train_enc = np.load(file=test_data_path + test_file)
        train_dec = np.concatenate([train_enc[6:, :, :, :], np.zeros(shape=(24, 24, 72, 4), dtype=np.float)])
        # train_dec = np.concatenate([train_enc[:, :, :, :], np.zeros(shape=(24, 24, 72, 4), dtype=np.float)])
        month_enc = np.array([[(month % 12) + 1 for month in range(start - 1, start + 11)]])
        month_dec = np.array([[(month % 12) + 1 for month in range(start + 5, start + 35)]])
        # month_dec = np.array([[(month % 12) + 1 for month in range(start - 1, start + 35)]])

        outputs = model(inputs=[train_enc, train_dec, month_enc, month_dec])
        treat_outputs = outputs[0, -24:, 0]

        np.save(file=result_save_path + test_file, arr=treat_outputs.numpy())

    make_zip(source_dir=result_save_path, output_filename="result.zip")


def main() -> NoReturn:
    """TensorFlow版transformer执行器入口
    """
    parser = ArgumentParser(description="informer", )
    parser.add_argument("--pre_epochs", default=20, type=int, required=False, help="训练步数")
    parser.add_argument("--epochs", default=10, type=int, required=False, help="训练步数")
    parser.add_argument("--enc_num_layers", default=3, type=int, required=False, help="encoder和decoder的内部层数")
    parser.add_argument("--dec_num_layers", default=2, type=int, required=False, help="encoder和decoder的内部层数")
    parser.add_argument("--num_heads", default=8, type=int, required=False, help="头注意力数量")
    parser.add_argument("--units", default=1024, type=int, required=False, help="隐藏层单元数")
    parser.add_argument("--dropout", default=0.1, type=float, required=False, help="dropout")
    parser.add_argument("--embedding_dim", default=512, type=int, required=False, help="嵌入层维度大小")
    parser.add_argument("--batch_size", default=1, type=int, required=False, help="batch大小")
    parser.add_argument("--buffer_size", default=10000, type=int, required=False, help="Dataset加载缓冲大小")
    parser.add_argument("--checkpoint_save_size", default=20, type=int, required=False, help="单轮训练中检查点保存数量")
    parser.add_argument("--checkpoint_save_freq", default=1, type=int, required=False, help="检查点保存频率")

    parser.add_argument("--checkpoint_dir", default="checkpoint/",
                        type=str, required=False, help="")
    parser.add_argument("--save_dir", default="user_data/train/", type=str, required=False, help="")
    parser.add_argument("--result_save_path", default="result/", type=str, required=False, help="")
    parser.add_argument("--save_pairs", default="user_data/pairs.txt", type=str, required=False, help="")
    parser.add_argument("--save_soda_pairs", default="user_data/soda_pairs.txt", type=str, required=False, help="")
    parser.add_argument("--save_cmip_pairs", default="user_data/cmip_pairs.txt", type=str, required=False, help="")

    parser.add_argument("--soda_train_data_path", default="tcdata/enso_round1_train_20210201/SODA_train.nc",
                        type=str, required=False, help="")
    parser.add_argument("--soda_label_data_path", default="tcdata/enso_round1_train_20210201/SODA_label.nc",
                        type=str, required=False, help="")
    parser.add_argument("--cmip_train_data_path", default="tcdata/enso_round1_train_20210201/CMIP_train.nc",
                        type=str, required=False, help="")
    parser.add_argument("--cmip_label_data_path", default="tcdata/enso_round1_train_20210201/CMIP_label.nc",
                        type=str, required=False, help="")
    parser.add_argument("--test_data_path", default="tcdata/enso_final_test_data_B/", type=str, required=False, help="")

    options = parser.parse_args()

    # print("正在处理soda语料")
    # preprocess_soda(
    #     train_data_path=options.soda_train_data_path, label_data_path=options.soda_label_data_path,
    #     save_pairs=options.save_soda_pairs, save_dir=options.save_dir
    # )
    # print("正在处理cmip语料")
    # preprocess_cmip(
    #     train_data_path=options.cmip_train_data_path, label_data_path=options.cmip_label_data_path,
    #     save_pairs=options.save_cmip_pairs, save_dir=options.save_dir
    # )
    #
    # print("加载模型中")
    informer_model = informer(
        embedding_dim=options.embedding_dim, enc_num_layers=options.enc_num_layers,
        dec_num_layers=options.dec_num_layers, batch_size=options.batch_size,
        num_heads=options.num_heads, dropout=options.dropout
    )
    #
    checkpoint_manager = load_checkpoint(checkpoint_dir=options.checkpoint_dir,
                                         checkpoint_save_size=options.checkpoint_save_size, model=informer_model)
    #
    # print("正在cmip预训练")
    # cmip_train_dataset, _ = load_dataset(pairs_path=options.save_cmip_pairs, batch_size=options.batch_size,
    #                                      buffer_size=options.buffer_size)
    # _ = train(model=informer_model, checkpoint=checkpoint_manager, batch_size=options.batch_size,
    #           epochs=options.pre_epochs, train_dataset=cmip_train_dataset, valid_dataset=None,
    #           checkpoint_save_freq=options.checkpoint_save_freq)
    #
    # print("正在进行soda微调训练")
    # soda_train_dataset, soda_valid_dataset = load_dataset(
    #     pairs_path=options.save_cmip_pairs, batch_size=options.batch_size, buffer_size=options.buffer_size
    # )
    # _ = train(model=informer_model, checkpoint=checkpoint_manager, batch_size=options.batch_size,
    #           epochs=options.epochs, train_dataset=soda_train_dataset, valid_dataset=soda_valid_dataset,
    #           checkpoint_save_freq=options.checkpoint_save_freq)

    # print("正在预测中")
    inference(model=informer_model, result_save_path=options.result_save_path, test_data_path=options.test_data_path)


if __name__ == "__main__":
    main()
