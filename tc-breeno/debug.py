from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jieba
import tensorflow as tf
import numpy as np
import collections
from code.tools import load_dataset
from code.tools import load_raw_dataset
from code.tools import load_tokenizer

if __name__ == "__main__":
    # tokenizer = preprocess_dict(data_dir="./tcdata/", dict_path="./user_data/dict.json")
    # exit(0)
    # dataset = load_dataset(record_path="./user_data/train.tfrecord", batch_size=6, buffer_size=100000)

    # ===================字典验证===================
    # tokenizer = load_tokenizer(dict_path="./user_data/dict.json")
    # print(tokenizer.word_index)
    # tokenizer.fit_on_sequences()

    # count = 0
    # for batch, (first_queries, second_queries, labels) in enumerate(dataset):
    #     print(first_queries)
    #     print(second_queries)
    #     print(labels)
    #     count += 1
    #     if count == 3:
    #         break

    # =================dataset验证===================
    # train_dataset = load_dataset(record_path="./user_data/train.tfrecord", batch_size=1, buffer_size=100000, data_type="valid")
    # for i in range(10):
    #     for (batch, ba) in enumerate(train_dataset.take(1)):
    #         print("==========================================")
    #         print(ba)

    # ===========统计数据信息====================
    # files = ["./tcdata/gaiic_track3_round1_train_20210220.tsv", "./tcdata/gaiic_track3_round1_testA_20210220v2.tsv"]
    # imax, imin, imean, count = 0, 0, 3.5, 0
    # for file_name in files:
    #     with open(file_name, "r", encoding="utf-8") as file:
    #         for line in file:
    #             line = line.strip().strip("\n").split("\t")
    #
    #             first = len((" ".join(jieba.cut(line[0]))).split(" "))
    #             second = len((" ".join(jieba.cut(line[1]))).split(" "))
    #
    #             imax = max(imax, first, second)
    #             imin = min(imin, first, second)
    #             imean = (imean + (first + second) / 2) / 2
    #
    #             count += 1
    #             if count % 1000 == 0:
    #                 print("\r已读取 {} 条query-pairs".format(count), end="", flush=True)
    #
    # print(imax, imin, imean) # 65 0 4.236324119718941
    # arr = np.array([], np.float)

    # ==============统计数据信息2.0===============
    # files = ["./tcdata/gaiic_track3_round1_train_20210228.tsv", "./tcdata/gaiic_track3_round1_testA_20210228.tsv"]
    # imax, imin, imean, count, amax = 0, 0, 3.5, 0, 0
    # total = collections.Counter()
    # all = []
    # for file_name in files:
    #     with open(file_name, "r", encoding="utf-8") as file:
    #         for line in file:
    #             line = line.strip().strip("\n").split("\t")
    #
    #             first_query = line[0].split(" ")
    #             second_query = line[1].split(" ")
    #             first = len(first_query)
    #             second = len(second_query)
    #
    #             for num in first_query:
    #                 amax = max(amax, int(num))
    #
    #             for num in second_query:
    #                 amax = max(amax, int(num))
    #
    #             total[first] += 1
    #             total[second] += 1
    #
    #             imax = max(imax, first, second)
    #             imin = min(imin, first, second)
    #             imean = (imean + (first + second) / 2) / 2
    #             all.append(first_query)
    #             all.append(second_query)
    #
    # count += 1
    # if count % 1000 == 0:
    #     print("\r已读取 {} 条query-pairs".format(count), end="", flush=True)
    #
    # tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", split=" ", oov_token="<unk>")
    # # print(all)
    # tokenizer.fit_on_sequences(all)
    # print(tokenizer.index_docs)
    #
    # print(imax, imin, imean, amax) # 79 0 4.900125129150295 21962
    # print(total)
    # path = "./tcdata/gaiic_track3_round1_train_20210228.tsv"
    # dataset = load_raw_dataset(data_path=path, max_sentence_length=40, batch_size=1, buffer_size=100000, data_type="train")
    # for i in range(6):
    #     for (batch, (queries, segments, outputs, label)) in enumerate(dataset.take(2)):
    #         print("==========================================")
    #         print(queries)
    #         print(segments)
    #         print(outputs)
    # y_true = [1, 2]
    # y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    # # Using 'auto'/'sum_over_batch_size' reduction type.
    # scce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    # scce1 = tf.keras.losses.SparseCategoricalCrossentropy()
    # print(scce(y_true, y_pred).numpy())
    # print(tf.multiply(scce(y_true, y_pred), tf.constant([0., 1.])))
    # print(scce1(y_true, y_pred).numpy())
    # a = tf.constant([1,2,3])
    # b = tf.constant([2,4,6])
    # print(a / b)

    # files = ["./tcdata/gaiic_track3_round1_train_20210228.tsv", "./tcdata/gaiic_track3_round1_testA_20210228.tsv"]
    # from code.preprocess import slice_neg_pos_data
    #
    # slice_neg_pos_data("./tcdata/gaiic_track3_round1_train_20210228.tsv", "./user_data/test.txt", if_self=False)
    # files = ["./tcdata/gaiic_track3_round1_train_20210228.tsv", "./tcdata/gaiic_track3_round1_testB_20210317.tsv"]
    # total = collections.Counter()
    # ma = 0
    # for file_name in files:
    #     with open(file_name, "r", encoding="utf-8") as file:
    #         for line in file:
    #             line = line.strip().strip("\n").split("\t")
    #
    #             first_query = line[0].split(" ")
    #             second_query = line[1].split(" ")
    #
    #             for num in first_query:
    #                 total[num] += 1
    #                 ma = max(ma, int(num))
    #             for num in second_query:
    #                 total[num] += 1
    #                 ma = max(ma, int(num))
    # print(total)
    # print(len(total))
    # print(ma) # 21963
    # sentence_to_vec方法就是将句子转换成对应向量的核心方法
    # def sentence_to_vec(sentence_list: List[Sentence], embedding_size: int, a: float = 1e-3):
    #     sentence_set = []
    #     for sentence in sentence_list:
    #         vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
    #         sentence_length = sentence.len()
    #         # 这个就是初步的句子向量的计算方法
    #         #################################################
    #         for word in sentence.word_list:
    #             a_value = a / (a + get_word_frequency(word.text))  # smooth inverse frequency, SIF
    #             vs = np.add(vs, np.multiply(a_value, word.vector))  # vs += sif * word_vector
    #
    #         vs = np.divide(vs, sentence_length)  # weighted average
    #         sentence_set.append(vs)  # add to our existing re-calculated set of sentences
    #     #################################################
    #     # calculate PCA of this sentence set,计算主成分
    #     pca = PCA()
    #     # 使用PCA方法进行训练
    #     pca.fit(np.array(sentence_set))
    #     # 返回具有最大方差的的成分的第一个,也就是最大主成分,
    #     # components_也就是特征个数/主成分个数,最大的一个特征值
    #     u = pca.components_[0]  # the PCA vector
    #     # 构建投射矩阵
    #     u = np.multiply(u, np.transpose(u))  # u x uT
    #     # judge the vector need padding by wheather the number of sentences less than embeddings_size
    #     # 判断是否需要填充矩阵,按列填充
    #     if len(u) < embedding_size:
    #         for i in range(embedding_size - len(u)):
    #             # 列相加
    #             u = np.append(u, 0)  # add needed extension for multiplication below
    #
    #     # resulting sentence vectors, vs = vs -u x uT x vs
    #     sentence_vecs = []
    #     for vs in sentence_set:
    #         sub = np.multiply(u, vs)
    #         sentence_vecs.append(np.subtract(vs, sub))
    #     return sentence_vecs

    with open("./user_data/train.tsv", "a", encoding="utf-8") as combine_file, open(
            "./user_data/result1.tsv", "r", encoding="utf-8") as result_file, open(
        "./tcdata/gaiic_track3_round1_testA_20210228.tsv", "r", encoding="utf-8") as label_file:
        for result, label in zip(result_file, label_file):
            label = label.strip().strip("\n").split("\t")
            result = result.strip().strip("\n").split("\t")

            if 0.15 < float(result[0]) < 0.85:
                continue

            label.append("1" if float(result[0]) >= 0.9 else "0")
            combine_file.write("\t".join(label) + "\n")

    with open("./user_data/train.tsv", "a", encoding="utf-8") as combine_file, open(
            "./user_data/result.tsv", "r", encoding="utf-8") as result_file, open(
        "./tcdata/gaiic_track3_round1_testB_20210317.tsv", "r", encoding="utf-8") as label_file:
        for result, label in zip(result_file, label_file):
            label = label.strip().strip("\n").split("\t")
            result = result.strip().strip("\n").split("\t")

            if 0.15 < float(result[0]) < 0.85:
                continue

            label.append("1" if float(result[0]) >= 0.9 else "0")
            combine_file.write("\t".join(label) + "\n")
