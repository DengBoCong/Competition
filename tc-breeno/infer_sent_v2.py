from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
from code.infer_sent_model import infer_sent
from code.preprocess import slice_data
from code.preprocess import slice_neg_pos_data
from code.tools import load_checkpoint
from code.infer_sent_modules import evaluate
from code.infer_sent_modules import inference
from code.infer_sent_modules import train
from typing import NoReturn


def main() -> NoReturn:
    parser = ArgumentParser(description="执行器")
    parser.add_argument("--act", default="preprocess", type=str, required=False, help="执行模式")
    parser.add_argument("--vocab_size", default=20601, type=int, required=False, help="词汇量大小")
    parser.add_argument("--epochs", default=100, type=int, required=False, help="训练周期")
    parser.add_argument("--num_layers", default=4, type=int, required=False, help="block层数")
    parser.add_argument("--units", default=1024, type=int, required=False, help="单元数")
    parser.add_argument("--embedding_dim", default=768, type=int, required=False, help="词嵌入大小")
    parser.add_argument("--num_heads", default=12, type=int, required=False, help="注意力头数")
    parser.add_argument("--dropout", default=0.1, type=float, required=False, help="采样率")
    parser.add_argument("--batch_size", default=512, type=int, required=False, help="batch大小")
    parser.add_argument("--buffer_size", default=100000, type=int, required=False, help="缓冲区大小")
    parser.add_argument("--max_sentence_length", default=20, type=int, required=False, help="最大句子序列长度")
    parser.add_argument("--checkpoint_save_size", default=100, type=int, required=False, help="最大保存检查点数量")
    parser.add_argument("--train_data_size", default=0, type=int, required=False, help="训练数据大小")
    parser.add_argument("--valid_data_size", default=0, type=int, required=False, help="验证数据大小")
    parser.add_argument("--max_train_steps", default=-1, type=int, required=False, help="最大训练数据量，-1为全部")
    parser.add_argument("--checkpoint_save_freq", default=1, type=int, required=False, help="检查点保存频率")
    parser.add_argument("--raw_train_data_path", default="./tcdata/gaiic_track3_round1_train_20210228.tsv", type=str,
                        required=False, help="原始训练数据相对路径")
    parser.add_argument("--raw_test_data_path", default="./tcdata/gaiic_track3_round1_testA_20210228.tsv", type=str,
                        required=False, help="原始测试数据相对路径")
    parser.add_argument("--slice_train_data_path", default="./user_data/slice_train.tsv", type=str,
                        required=False, help="训练数据相对路径")
    parser.add_argument("--train_data_path", default="./user_data/train.tsv", type=str,
                        required=False, help="训练数据相对路径")
    parser.add_argument("--valid_data_path", default="./user_data/valid.tsv", type=str,
                        required=False, help="验证数据相对路径")
    parser.add_argument("--checkpoint_dir", default="./user_data/checkpoint/", type=str, required=False,
                        help="验证数据的TFRecord格式保存相对路径")
    parser.add_argument("--result_save_path", default="./user_data/result.tsv", type=str, required=False,
                        help="测试数据的结果文件")

    options = parser.parse_args()
    match_model = infer_sent(
        vocab_size=options.vocab_size, num_layers=options.num_layers, units=options.units,
        embedding_dim=options.embedding_dim, num_heads=options.num_heads, dropout=options.dropout
    )

    checkpoint_manager = load_checkpoint(checkpoint_dir=options.checkpoint_dir, execute_type=options.act,
                                         checkpoint_save_size=options.checkpoint_save_size, model=match_model)

    if options.act == "preprocess":
        print("正在切分训练数据")
        slice_data(data_path=options.raw_train_data_path, first_split_path=options.slice_train_data_path,
                   second_split_path=options.valid_data_path, split=0.1)
        print("正在增强数据")
        slice_neg_pos_data(data_path=options.slice_train_data_path,
                           save_path=options.train_data_path, if_self=False)
    elif options.act == "train":
        history = train(
            model=match_model, checkpoint=checkpoint_manager, batch_size=options.batch_size,
            buffer_size=options.buffer_size, epochs=options.epochs, embedding_dim=options.embedding_dim,
            train_data_path=options.train_data_path, valid_data_path=options.valid_data_path,
            max_sentence_length=options.max_sentence_length, max_train_steps=-1,
            checkpoint_save_freq=options.checkpoint_save_freq
        )
    elif options.act == "evaluate":
        evaluate(model=match_model, batch_size=options.batch_size, buffer_size=options.buffer_size,
                 valid_data_path=options.valid_data_path, max_sentence_length=options.max_sentence_length)
    elif options.act == "inference":
        inference(model=match_model, result_save_path=options.result_save_path,
                  raw_data_path=options.raw_test_data_path, max_sentence=options.max_sentence_length)
    else:
        parser.error(message="")


if __name__ == "__main__":
    main()
