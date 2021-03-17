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

import gc
from argparse import ArgumentParser
from code.preprocess import preprocess_soda
from code.preprocess import preprocess_cmip
from code.tools import load_checkpoint
from code.model import informer
from typing import NoReturn
from code.modules import train
from code.modules import evaluate
from code.modules import inference


def tf_transformer() -> NoReturn:
    """TensorFlow版transformer执行器入口
    """
    parser = ArgumentParser(description="informer", )
    parser.add_argument("--act", default="train", type=str, required=False, help="执行类型")
    parser.add_argument("--enc_num_layers", default=3, type=int, required=False, help="encoder和decoder的内部层数")
    parser.add_argument("--dec_num_layers", default=2, type=int, required=False, help="encoder和decoder的内部层数")
    parser.add_argument("--num_heads", default=8, type=int, required=False, help="头注意力数量")
    parser.add_argument("--units", default=1024, type=int, required=False, help="隐藏层单元数")
    parser.add_argument("--dropout", default=0.05, type=float, required=False, help="dropout")
    parser.add_argument("--embedding_dim", default=512, type=int, required=False, help="嵌入层维度大小")
    parser.add_argument("--valid_data_file", default="", type=str, required=False, help="验证数据集路径")
    parser.add_argument("--valid_freq", default=5, type=int, required=False, help="验证频率")
    parser.add_argument("--checkpoint_save_freq", default=1, type=int, required=False, help="检查点保存频率")
    parser.add_argument("--checkpoint_save_size", default=5, type=int, required=False, help="单轮训练中检查点保存数量")
    parser.add_argument("--batch_size", default=1, type=int, required=False, help="batch大小")
    parser.add_argument("--buffer_size", default=10000, type=int, required=False, help="Dataset加载缓冲大小")
    parser.add_argument("--epochs", default=20, type=int, required=False, help="训练步数")
    parser.add_argument("--checkpoint_dir", default="./user_data/checkpoint/",
                        type=str, required=False, help="")
    parser.add_argument("--soda_train_data_path", default="./tcdata/enso_round1_train_20210201/SODA_train.nc",
                        type=str, required=False, help="")
    parser.add_argument("--soda_label_data_path", default="./tcdata/enso_round1_train_20210201/SODA_label.nc",
                        type=str, required=False, help="")
    parser.add_argument("--cmip_train_data_path", default="./tcdata/enso_round1_train_20210201/CMIP_train.nc",
                        type=str, required=False, help="")
    parser.add_argument("--cmip_label_data_path", default="./tcdata/enso_round1_train_20210201/CMIP_label.nc",
                        type=str, required=False, help="")
    parser.add_argument("--save_dir", default="./user_data/train/", type=str, required=False, help="")
    parser.add_argument("--result_save_path", default="./user_data/result/", type=str, required=False, help="")
    parser.add_argument("--test_data_path", default="./tcdata/test/", type=str, required=False, help="")
    parser.add_argument("--soda_save_pairs", default="./user_data/soda_pairs.txt", type=str, required=False, help="")
    parser.add_argument("--cmip_save_pairs", default="./user_data/cmip_pairs.txt", type=str, required=False, help="")

    options = parser.parse_args()

    model = informer(
        embedding_dim=options.embedding_dim, enc_num_layers=options.enc_num_layers,
        dec_num_layers=options.dec_num_layers, batch_size=options.batch_size,
        num_heads=options.num_heads, dropout=options.dropout
    )

    checkpoint_manager = load_checkpoint(checkpoint_dir=options.checkpoint_dir, execute_type=options.act,
                                         checkpoint_save_size=options.checkpoint_save_size, model=model)

    if options.act == "preprocess":
        # preprocess_soda(
        #     train_data_path=options.soda_train_data_path, label_data_path=options.soda_label_data_path,
        #     save_pairs=options.soda_save_pairs, save_dir=options.save_dir
        # )
        # gc.collect()
        preprocess_cmip(
            train_data_path=options.cmip_train_data_path, label_data_path=options.cmip_label_data_path,
            save_pairs=options.cmip_save_pairs, save_dir=options.save_dir
        )
    elif options.act == "train":

        history = train(model=model, checkpoint=checkpoint_manager, batch_size=options.batch_size,
                        epochs=options.epochs, train_dataset=train_dataset, valid_dataset=valid_dataset,
                        checkpoint_save_freq=options.checkpoint_save_freq)
    elif options.act == "evaluate":
        train_dataset, valid_dataset = preprocess_soda(
            train_data_path=options.soda_train_data_path, label_data_path=options.soda_label_data_path,
            save_dir=options.save_dir, data_type="soda", batch_size=options.batch_size, buffer_size=options.buffer_size
        )
        evaluate(model=model, batch_size=options.batch_size, dataset=valid_dataset)
    elif options.act == "inference":
        inference(model=model, result_save_path=options.result_save_path, test_data_path=options.test_data_path)
    else:
        parser.error(message="")


if __name__ == "__main__":
    tf_transformer()
