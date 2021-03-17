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
import zipfile
import numpy as np
import tensorflow as tf
from model import informer
from argparse import ArgumentParser
from typing import Any
from typing import AnyStr
from typing import NoReturn


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
        start = int(os.path.splitext(name)[0].split("_")[2])

        train_enc = np.load(file=test_data_path + test_file)
        train_dec = np.concatenate([train_enc[6:, :, :, :], np.zeros(shape=(24, 24, 72, 4), dtype=np.float)])
        month_enc = np.array([[(month % 12) + 1 for month in range(start - 1, start + 11)]])
        month_dec = np.array([[(month % 12) + 1 for month in range(start + 5, start + 35)]])

        outputs = model(inputs=[train_enc, train_dec, month_enc, month_dec])
        treat_outputs = outputs[0, -24:, 0]

        np.save(file=result_save_path + test_file, arr=treat_outputs.numpy())

    make_zip(source_dir=result_save_path, output_filename="result.zip")


def main() -> NoReturn:
    """TensorFlow版transformer执行器入口
    """
    parser = ArgumentParser(description="informer", )
    parser.add_argument("--enc_num_layers", default=3, type=int, required=False, help="encoder和decoder的内部层数")
    parser.add_argument("--dec_num_layers", default=2, type=int, required=False, help="encoder和decoder的内部层数")
    parser.add_argument("--num_heads", default=8, type=int, required=False, help="头注意力数量")
    parser.add_argument("--units", default=1024, type=int, required=False, help="隐藏层单元数")
    parser.add_argument("--dropout", default=0.05, type=float, required=False, help="dropout")
    parser.add_argument("--embedding_dim", default=512, type=int, required=False, help="嵌入层维度大小")
    parser.add_argument("--batch_size", default=1, type=int, required=False, help="batch大小")
    parser.add_argument("--checkpoint_dir", default="checkpoint/",
                        type=str, required=False, help="")
    parser.add_argument("--save_dir", default="user_data/train/", type=str, required=False, help="")
    parser.add_argument("--result_save_path", default="result/", type=str, required=False, help="")
    parser.add_argument("--test_data_path", default="tcdata/enso_round1_test_20210201/", type=str, required=False, help="")

    options = parser.parse_args()
    model = informer(
        embedding_dim=options.embedding_dim, enc_num_layers=options.enc_num_layers,
        dec_num_layers=options.dec_num_layers, batch_size=options.batch_size,
        num_heads=options.num_heads, dropout=options.dropout
    )

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(options.checkpoint_dir)).expect_partial()

    inference(model=model, result_save_path=options.result_save_path, test_data_path=options.test_data_path)


if __name__ == "__main__":
    main()
