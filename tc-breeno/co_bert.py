from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import *
import tensorflow as tf
from argparse import ArgumentParser
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import TFBertForMaskedLM
from transformers import AutoTokenizer, TFAutoModelForMaskedLM
from transformers import TFBertModel
from code.co_modules import bert_model
from code.co_modules import train
from code.co_modules import evaluate
from code.co_modules import inference
from code.tools import load_checkpoint


def main() -> NoReturn:
    parser = ArgumentParser(description="执行器")
    parser.add_argument("--act", default="preprocess", type=str, required=False, help="执行模式")
    parser.add_argument("--vocab_size", default=6932, type=int, required=False, help="词汇量大小")
    parser.add_argument("--epochs", default=50, type=int, required=False, help="训练周期")
    parser.add_argument("--num_layers", default=12, type=int, required=False, help="block层数")
    parser.add_argument("--units", default=1024, type=int, required=False, help="单元数")
    parser.add_argument("--first_kernel_size", default=3, type=int, required=False, help="第一个卷积核大小")
    parser.add_argument("--second_kernel_size", default=3, type=int, required=False, help="第二个卷积核大小")
    parser.add_argument("--first_strides_size", default=3, type=int, required=False, help="第一个卷积步幅大小")
    parser.add_argument("--second_strides_size", default=3, type=int, required=False, help="第二个卷积步幅大小")
    parser.add_argument("--first_output_dim", default=32, type=int, required=False, help="第一个卷积输出通道数")
    parser.add_argument("--second_output_dim", default=16, type=int, required=False, help="第二个卷积输出通道数")
    parser.add_argument("--embedding_dim", default=768, type=int, required=False, help="词嵌入大小")
    parser.add_argument("--num_heads", default=12, type=int, required=False, help="注意力头数")
    parser.add_argument("--dropout", default=0.1, type=float, required=False, help="采样率")
    parser.add_argument("--batch_size", default=32, type=int, required=False, help="batch大小")
    parser.add_argument("--buffer_size", default=100000, type=int, required=False, help="缓冲区大小")
    parser.add_argument("--max_sentence_length", default=32, type=int, required=False, help="最大句子序列长度")
    parser.add_argument("--checkpoint_save_size", default=10, type=int, required=False, help="最大保存检查点数量")
    parser.add_argument("--train_data_size", default=0, type=int, required=False, help="训练数据大小")
    parser.add_argument("--valid_data_size", default=0, type=int, required=False, help="验证数据大小")
    parser.add_argument("--max_train_steps", default=-1, type=int, required=False, help="最大训练数据量，-1为全部")
    parser.add_argument("--checkpoint_save_freq", default=1, type=int, required=False, help="检查点保存频率")
    parser.add_argument("--data_dir", default="./tcdata/", type=str, required=False, help="原始数据保存目录")
    parser.add_argument("--raw_train_data_path", default="./tcdata/gaiic_track3_round1_train_20210228.tsv", type=str,
                        required=False, help="原始训练数据相对路径")
    parser.add_argument("--raw_test_data_path", default="./tcdata/gaiic_track3_round1_testA_20210228.tsv", type=str,
                        required=False, help="原始测试数据相对路径")
    parser.add_argument("--train_data_path", default="./user_data/train.tsv", type=str, required=False, help="训练数据相对路径")
    parser.add_argument("--valid_data_path", default="./user_data/valid.tsv", type=str, required=False, help="验证数据相对路径")
    parser.add_argument("--train_record_data_path", default="./user_data/train.tfrecord", type=str, required=False,
                        help="训练数据的TFRecord格式保存相对路径")
    parser.add_argument("--valid_record_data_path", default="./user_data/valid.tfrecord", type=str, required=False,
                        help="验证数据的TFRecord格式保存相对路径")
    parser.add_argument("--test_record_data_path", default="./user_data/test.tfrecord", type=str, required=False,
                        help="测试数据的TFRecord格式保存相对路径")
    parser.add_argument("--checkpoint_dir", default="./user_data/checkpointv1/", type=str, required=False,
                        help="验证数据的TFRecord格式保存相对路径")
    parser.add_argument("--result_save_path", default="./user_data/result.tsv", type=str, required=False,
                        help="测试数据的结果文件")
    parser.add_argument("--config_path", default="./tcdata/bert/config.json", type=str, required=False,
                        help="配置文件路径")
    parser.add_argument("--bert_path", default="./tcdata/bert/tf_model.h5", type=str, required=False,
                        help="Bert路径")
    parser.add_argument("--dict_path", default="./tcdata/bert/vocab.txt", type=str, required=False, help="字典保存路径")

    options = parser.parse_args()
    # bert_model = model(vocab_size=)
    # model_path = "../tcdata/bert/"
    # tokenizer = BertTokenizer.from_pretrained("../tcdata/bert/vocab.txt")
    model_config = BertConfig.from_pretrained("./tcdata/bert/config.json")
    model_config.output_attentions = False
    model_config.output_hidden_states = False
    model_config.use_cache = False
    # model = TFBertForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_path, from_pt=False,
    #                                           config=model_config, cache_dir="../user_data/temp")
    # model.resize_token_embeddings(len(tokenizer))

    # tokenizer = AutoTokenizer.from_pretrained("./tcdata/bert")
    bert = TFAutoModelForMaskedLM.from_pretrained("./tcdata/bert", config=model_config, cache_dir="../user_data/temp")
    # token = tokenizer.encode("生活的真谛是[MASK]。")
    # print(tokenizer.decode(token))
    # input = tf.convert_to_tensor([token])
    # print(input)
    # outputs = bert(input)[0]
    # print(tokenizer.decode(tf.argmax(outputs[0],axis=-1)))
    # exit(0)

    # model_config = BertConfig.from_pretrained(options.config_path)
    # bert = TFBertModel.from_pretrained(pretrained_model_name_or_path=options.bert_path, from_pt=False,
    #                                    config=model_config, cache_dir="../user_data/temp")
    # bert.resize_token_embeddings(new_num_tokens=options.vocab_size)

    model = bert_model(vocab_size=options.vocab_size, bert=bert)

    checkpoint_manager = load_checkpoint(checkpoint_dir=options.checkpoint_dir, execute_type=options.act,
                                         checkpoint_save_size=options.checkpoint_save_size, model=model)

    if options.act == "train":
        history = train(
            model=model, checkpoint=checkpoint_manager, batch_size=options.batch_size, buffer_size=options.buffer_size,
            epochs=options.epochs, train_data_path=options.raw_train_data_path,
            test_data_path=options.raw_test_data_path, dict_path=options.dict_path,
            max_sentence_length=options.max_sentence_length, checkpoint_save_freq=options.checkpoint_save_freq
        )
    elif options.act == "evaluate":
        pass
    elif options.act == "inference":
        pass
    else:
        parser.error(message="")


if __name__ == '__main__':
    # from code.co_modules import preprocess
    # from code.co_modules import statistics
    #
    # data_v, train_data_v, valid_data_v, test_data_v = preprocess(
    #     train_data_path="../tcdata/gaiic_track3_round1_train_20210228.tsv",
    #     test_data_path="../tcdata/gaiic_track3_round1_testA_20210228.tsv")
    # tokens, keep_tokens = statistics(min_freq=5, dict_path="../tcdata/bert/vocab.txt", data=data_v + test_data_v)
    #
    # for index, (first_query, second_query, label) in enumerate(data_v):
    #     print(first_query, second_query, label)
    #     va
    # exit()
    # model_config.output_attentions = False
    # model_config.output_hidden_states = False
    # model_config.use_cache = True
    # #

    # 3333333333333333333333333333333333333333
    # tokenizer = BertTokenizer.from_pretrained("./tcdata/bert/vocab.txt")
    # model_config = BertConfig.from_pretrained("./tcdata/bert/config.json")
    # model_config.output_attentions = False
    # model_config.output_hidden_states = False
    # model_config.use_cache = True
    # bert_model = TFBertModel.from_pretrained(pretrained_model_name_or_path="./tcdata/bert/tf_model.h5", from_pt=False,
    #                                          config=model_config, cache_dir="./user_data/temp")
    # model = TFBertForMaskedLM(config=model_config)
    # model.bert = bert_model
    # model.resize_token_embeddings(len(tokenizer))
    # model = TFBertForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_path, from_pt=False,
    #                                           config=model_config, cache_dir="../user_data/temp")
    # model.resize_token_embeddings(len(tokenizer))
    #

    # inputs = tokenizer("中国的首都是[MASK]", return_tensors="tf")
    # inputs["labels"] = tokenizer("中国的首都是北京", return_tensors="tf")["input_ids"]
    # inputs = tokenizer.encode("wiki系统属于一种人类[MASK][MASK]的网络系统", return_tensors="tf")
    # print(tokenizer.tokenize("中国的首都是[MASK]"))
    # outputs = bert_model(inputs)
    # print(outputs.last_hidden_state.shape)
    # o1 = tf.argmax(outputs.logits[0], axis=1)
    # print(o1)
    # print(tokenizer.decode(o1))

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
    # from code.co_modules import preprocess
    # from code.co_modules import statistics
    # from code.co_modules import sample_convert
    #
    # max_sentence_length = 45
    #
    # data_v, train_data_v, valid_data_v, test_data_v = preprocess(
    #     train_data_path="../tcdata/gaiic_track3_round1_train_20210228.tsv",
    #     test_data_path="../tcdata/gaiic_track3_round1_testA_20210228.tsv")
    # tokens, keep_tokens = statistics(min_freq=3, dict_path="../tcdata/bert/vocab.txt", data=data_v + test_data_v)
    #
    # print(len(keep_tokens))
    # exit(0)
    #
    # valid_queries = tf.constant(value=0, shape=(1, max_sentence_length), dtype=tf.int32)
    # valid_segments = tf.constant(value=0, shape=(1, max_sentence_length), dtype=tf.int32)
    # valid_labels = []
    # count = 0
    # for index, (first_query, second_query, label) in enumerate(valid_data_v):
    #     print(first_query, second_query, label)
    #     text1_ids = [keep_tokens[tokens.get(t, 1)] for t in first_query]
    #     text2_ids = [keep_tokens[tokens.get(t, 1)] for t in second_query]
    #
    #     queries = [2] + text1_ids + [3] + text2_ids + [3]
    #     segments = [0] * len(queries)
    #
    #     queries = tf.keras.preprocessing.sequence.pad_sequences([queries], max_sentence_length, padding="post")
    #     segments = tf.keras.preprocessing.sequence.pad_sequences([segments], max_sentence_length, padding="post")
    #
    #     valid_queries = tf.concat([valid_queries, queries], axis=0)
    #     valid_segments = tf.concat([valid_segments, segments], axis=0)
    #     valid_labels.append(label)
    #
    #     count += 1
    #     if count == 3:
    #         print(valid_queries, valid_segments, valid_labels)
    #         exit(0)
    main()
    # a = tf.constant(3.)
    # b = tf.constant(6.)
    # print(tf.divide(a, b))
    # print(tf.reduce_sum(a))
