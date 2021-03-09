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
"""transformer的实现执行器入口
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import tensorflow as tf
from argparse import ArgumentParser
from code.preprocess import preprocess_data_diff
from code.tools import load_checkpoint
from code.model import informer
from typing import NoReturn


def tf_transformer() -> NoReturn:
    """TensorFlow版transformer执行器入口
    """
    parser = ArgumentParser(description="transformer chatbot", )
    parser.add_argument("--config_file", default="", type=str, required=False, help="配置文件路径，为空则默认命令行，不为空则使用配置文件参数")
    parser.add_argument("--act", default="preprocess", type=str, required=False, help="执行类型")
    parser.add_argument("--num_layers", default=2, type=int, required=False, help="encoder和decoder的内部层数")
    parser.add_argument("--num_heads", default=8, type=int, required=False, help="头注意力数量")
    parser.add_argument("--units", default=512, type=int, required=False, help="隐藏层单元数")
    parser.add_argument("--dropout", default=0.1, type=float, required=False, help="dropout")
    parser.add_argument("--vocab_size", default=1500, type=int, required=False, help="词汇大小")
    parser.add_argument("--embedding_dim", default=256, type=int, required=False, help="嵌入层维度大小")
    parser.add_argument("--learning_rate_beta_1", default=0.9, type=int, required=False, help="一阶动量的指数加权平均权值")
    parser.add_argument("--learning_rate_beta_2", default=0.98, type=int, required=False, help="二阶动量的指数加权平均权值")
    parser.add_argument("--max_train_data_size", default=0, type=int, required=False, help="用于训练的最大数据大小")
    parser.add_argument("--max_valid_data_size", default=0, type=int, required=False, help="用于验证的最大数据大小")
    parser.add_argument("--max_sentence", default=40, type=int, required=False, help="单个序列的最大长度")
    parser.add_argument("--valid_data_file", default="", type=str, required=False, help="验证数据集路径")
    parser.add_argument("--valid_freq", default=5, type=int, required=False, help="验证频率")
    parser.add_argument("--checkpoint_save_freq", default=2, type=int, required=False, help="检查点保存频率")
    parser.add_argument("--checkpoint_save_size", default=3, type=int, required=False, help="单轮训练中检查点保存数量")
    parser.add_argument("--batch_size", default=1, type=int, required=False, help="batch大小")
    parser.add_argument("--buffer_size", default=20000, type=int, required=False, help="Dataset加载缓冲大小")
    parser.add_argument("--valid_data_split", default=0.2, type=float, required=False, help="从训练数据集中划分验证数据的比例")
    parser.add_argument("--epochs", default=5, type=int, required=False, help="训练步数")
    parser.add_argument("--checkpoint_dir", default="./user_data/checkpoint/",
                        type=str, required=False, help="")
    parser.add_argument("--soda_train_data_path", default="./tcdata/enso_round1_train_20210201/SODA_train.nc",
                        type=str, required=False, help="")
    parser.add_argument("--soda_label_data_path", default="./tcdata/enso_round1_train_20210201/SODA_label.nc",
                        type=str, required=False, help="")
    parser.add_argument("--save_dir", default="./user_data/train/",
                        type=str, required=False, help="")

    options = parser.parse_args()

    model = informer(embedding_dim=options.embedding_dim, num_layers=options.num_layers,
                     batch_size=options.batch_size, num_heads=options.num_heads, dropout=options.dropout)

    checkpoint_manager = load_checkpoint(checkpoint_dir=options.checkpoint_dir, execute_type=options.act,
                                         checkpoint_save_size=options.checkpoint_save_size, model=model)

    if options.act == "preprocess":
        preprocess_data_diff(train_data_path=options.soda_train_data_path, label_data_path=options.soda_label_data_path,
                             save_dir=options.save_dir, data_type="soda")
    elif options.act == "train":
        pass
    elif options.act == "evaluate":
        pass
    elif options.act == "inference":
        pass
    else:
        parser.error(message="")


if __name__ == "__main__":
    tf_transformer()
