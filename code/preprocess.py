import gc
import numpy as np
import tensorflow as tf
from netCDF4 import Dataset
from typing import *
from code.tools import process_train_pairs


def preprocess_cmip(train_data_path: AnyStr, label_data_path: AnyStr,
                    save_dir: AnyStr, save_pairs: AnyStr, split: Any = 6) -> NoReturn:
    """ 预处理数据集

    :param train_data_path: 训练集所在路径
    :param label_data_path: 标签数据集所在路径
    :param save_dir: 保存目录
    :param save_pairs: 处理好的文件路径pairs
    :param split: 对训练数据进行切分的起始位置
    :return: 无返回值
    """
    count = 0
    train_data = Dataset(filename=train_data_path, mode="r")
    label_data = Dataset(filename=label_data_path, mode="r")
    remain = [(0, 151, 1836, save_dir + "cmip1/"), (151, 302, 1836, save_dir + "cmip2/"),
              (302, 453, 1836, save_dir + "cmip3/"), (453, 604, 1836, save_dir + "cmip4/"),
              (604, 755, 1836, save_dir + "cmip5/"), (755, 906, 1836, save_dir + "cmip6/"),
              (906, 1057, 1836, save_dir + "cmip7/"), (1057, 1208, 1836, save_dir + "cmip8/"),
              (1208, 1359, 1836, save_dir + "cmip9/"), (1359, 1510, 1836, save_dir + "cmip10/"),
              (1510, 1661, 1836, save_dir + "cmip11/"), (1661, 1812, 1836, save_dir + "cmip12/"),
              (1812, 1963, 1836, save_dir + "cmip13/"), (1963, 2114, 1836, save_dir + "cmip14/"),
              (2114, 2265, 1836, save_dir + "cmip15/"), (2265, 2405, 1704, save_dir + "cmip16/"),
              (2405, 2545, 1704, save_dir + "cmip17/"), (2545, 2685, 1704, save_dir + "cmip18/"),
              (2685, 2825, 1704, save_dir + "cmip19/"), (2825, 2965, 1704, save_dir + "cmip20/"),
              (2965, 3105, 1704, save_dir + "cmip21/"), (3105, 3245, 1704, save_dir + "cmip22/"),
              (3245, 3385, 1704, save_dir + "cmip23/"), (3385, 3525, 1704, save_dir + "cmip24/"),
              (3525, 3665, 1704, save_dir + "cmip25/"), (3665, 3805, 1704, save_dir + "cmip26/"),
              (3805, 3945, 1704, save_dir + "cmip27/"), (3945, 4085, 1704, save_dir + "cmip28/"),
              (4085, 4225, 1704, save_dir + "cmip29/"), (4225, 4365, 1704, save_dir + "cmip30/"),
              (4365, 4505, 1704, save_dir + "cmip31/"), (4505, 4645, 1704, save_dir + "cmip32/")]

    for _, (begin, flag, all_years, depart_save_dir) in enumerate(remain):
        all_label_data = np.array(label_data["nino"][begin], dtype=np.float)
        enc_train_data = np.concatenate(
            [np.array(train_data["sst"][begin]).reshape([-1, 24, 72, 1]),
             np.array(train_data["t300"][begin]).reshape([-1, 24, 72, 1]),
             np.array(train_data["ua"][begin]).reshape([-1, 24, 72, 1]),
             np.array(train_data["va"][begin]).reshape([-1, 24, 72, 1])], axis=-1
        )
        begin += 1

        for i in range(begin, flag):
            arr = np.concatenate(
                [np.array(train_data["sst"][i]).reshape([-1, 24, 72, 1]),
                 np.array(train_data["t300"][i]).reshape([-1, 24, 72, 1]),
                 np.array(train_data["ua"][i]).reshape([-1, 24, 72, 1]),
                 np.array(train_data["va"][i]).reshape([-1, 24, 72, 1])], axis=-1
            )[-12:, :, :, :]

            enc_train_data = np.concatenate([enc_train_data, arr], axis=0)
            all_label_data = np.concatenate(
                [all_label_data, np.array(label_data["nino"][i], dtype=np.float)[-12:]], axis=-1)
            gc.collect()

        with open(save_pairs, "a", encoding="utf-8") as save_file:
            for start in range(all_years - 36):
                np.save(file=depart_save_dir + "train_enc_{}_{}_{}".format(start + 1, start + 1, start + 12),
                        arr=enc_train_data[start:start + 12, :, :, :])
                np.save(file=depart_save_dir + "train_dec_{}_{}_{}".format(start + 1, start + split + 1, start + 36),
                        arr=np.concatenate([enc_train_data[start + split:start + 12, :, :, :],
                                            np.zeros(shape=(24, 24, 72, 4), dtype=np.float)], axis=0))
                np.save(file=depart_save_dir + "month_enc_{}_{}_{}".format(start + 1, start + 1, start + 12),
                        arr=np.array([(month % 12) + 1 for month in range(start, start + 12)]))
                np.save(file=depart_save_dir + "month_dec_{}_{}_{}".format(start + 1, start + split + 1, start + 36),
                        arr=np.array([(month % 12) + 1 for month in range(start + split, start + 36)]))
                np.save(file=depart_save_dir + "label_{}_{}_{}".format(start + 1, start + 13, start + 36),
                        arr=all_label_data[start + 12:start + 36])

                save_file.write("{}\t{}\t{}\t{}\t{}\n".format(
                    depart_save_dir + "train_enc_{}_{}_{}.npy".format(start + 1, start + 1, start + 12),
                    depart_save_dir + "train_dec_{}_{}_{}.npy".format(start + 1, start + split + 1, start + 36),
                    depart_save_dir + "month_enc_{}_{}_{}.npy".format(start + 1, start + 1, start + 12),
                    depart_save_dir + "month_dec_{}_{}_{}.npy".format(start + 1, start + split + 1, start + 36),
                    depart_save_dir + "label_{}_{}_{}.npy".format(start + 1, start + 13, start + 36)
                ))

                count += 1
                if count % 100 == 0:
                    print("\r已生成 {} 条时间序列数据".format(count), end="", flush=True)


def preprocess_soda(train_data_path: AnyStr, label_data_path: AnyStr,
                    save_dir: AnyStr, save_pairs: AnyStr, split: Any = 6) -> NoReturn:
    """ 预处理数据集

    :param train_data_path: 训练集所在路径
    :param label_data_path: 标签数据集所在路径
    :param save_dir: 保存目录
    :param save_pairs: 处理好的文件路径pairs
    :param split: 对训练数据进行切分的起始位置
    :return: 无返回值
    """
    train_data = Dataset(filename=train_data_path, mode="r")
    label_data = Dataset(filename=label_data_path, mode="r")

    flag, all_years = 100, 1224

    all_label_data = np.array(label_data["nino"][0], dtype=np.float)
    enc_train_data = np.concatenate(
        [np.array(train_data["sst"][0]).reshape([-1, 24, 72, 1]),
         np.array(train_data["t300"][0]).reshape([-1, 24, 72, 1]),
         np.array(train_data["ua"][0]).reshape([-1, 24, 72, 1]),
         np.array(train_data["va"][0]).reshape([-1, 24, 72, 1])], axis=-1
    )

    for i in range(1, flag):
        arr = np.concatenate(
            [np.array(train_data["sst"][i]).reshape([-1, 24, 72, 1]),
             np.array(train_data["t300"][i]).reshape([-1, 24, 72, 1]),
             np.array(train_data["ua"][i]).reshape([-1, 24, 72, 1]),
             np.array(train_data["va"][i]).reshape([-1, 24, 72, 1])], axis=-1
        )[-12:, :, :, :]

        enc_train_data = np.concatenate([enc_train_data, arr], axis=0)
        all_label_data = np.concatenate(
            [all_label_data, np.array(label_data["nino"][i], dtype=np.float)[-12:]], axis=-1)
        gc.collect()

    count = 0
    # train_enc, train_dec, month_enc, month_dec, labels = [], [], [], [], []
    with open(save_pairs, "a", encoding="utf-8") as save_file:
        for start in range(all_years - 36):
            np.save(file=save_dir + "train_enc_{}_{}_{}".format(start + 1, start + 1, start + 12),
                    arr=enc_train_data[start:start + 12, :, :, :])
            np.save(file=save_dir + "train_dec_{}_{}_{}".format(start + 1, start + split + 1, start + 36),
                    arr=np.concatenate([enc_train_data[start + split:start + 12, :, :, :],
                                        np.zeros(shape=(24, 24, 72, 4), dtype=np.float)], axis=0))
            np.save(file=save_dir + "month_enc_{}_{}_{}".format(start + 1, start + 1, start + 12),
                    arr=np.array([(month % 12) + 1 for month in range(start, start + 12)]))
            np.save(file=save_dir + "month_dec_{}_{}_{}".format(start + 1, start + split + 1, start + 36),
                    arr=np.array([(month % 12) + 1 for month in range(start + split, start + 36)]))
            np.save(file=save_dir + "label_{}_{}_{}".format(start + 1, start + 13, start + 36),
                    arr=all_label_data[start + 12:start + 36])

            save_file.write("{}\t{}\t{}\t{}\t{}\n".format(
                save_dir + "train_enc_{}_{}_{}.npy".format(start + 1, start + 1, start + 12),
                save_dir + "train_dec_{}_{}_{}.npy".format(start + 1, start + split + 1, start + 36),
                save_dir + "month_enc_{}_{}_{}.npy".format(start + 1, start + 1, start + 12),
                save_dir + "month_dec_{}_{}_{}.npy".format(start + 1, start + split + 1, start + 36),
                save_dir + "label_{}_{}_{}.npy".format(start + 1, start + 13, start + 36)
            ))

            count += 1
            if count % 100 == 0:
                print("\r已生成 {} 条时间序列数据".format(count), end="", flush=True)

    # train_dataset = tf.data.Dataset.from_tensor_slices(
    #     (train_enc[:1000], train_dec[:1000], month_enc[:1000], month_dec[:1000], labels[:1000])
    # )
    # train_dataset = tf.data.Dataset.from_tensor_slices(
    #     (train_enc, train_dec, month_enc, month_dec, labels)
    # )
    # train_dataset = train_dataset.shuffle(
    #     buffer_size=buffer_size, reshuffle_each_iteration=True
    # ).prefetch(tf.data.experimental.AUTOTUNE)
    # train_dataset = train_dataset.map(
    #     process_train_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size, drop_remainder=True)

    # valid_dataset = tf.data.Dataset.from_tensor_slices(
    #     (train_enc[1000:], train_dec[1000:], month_enc[1000:], month_dec[1000:], labels[1000:])
    # )
    # valid_dataset = valid_dataset.map(
    #     process_train_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size, drop_remainder=True)

    # return train_dataset  # , valid_dataset

# def preprocess_soda_data_diff(train_data_path: AnyStr, label_data_path: AnyStr, split: Any = 6):
#     """ 预处理数据集
#
#     :param train_data_path: 训练集所在路径
#     :param label_data_path: 标签数据集所在路径
#     :param split: 对训练数据进行切分的起始位置
#     :return: 无返回值
#     """
#     train_data = Dataset(filename=train_data_path, mode="r")
#     label_data = Dataset(filename=label_data_path, mode="r")
#
#     labels = np.array(label_data["nino"][:], dtype=np.float)
#     sst = np.array(train_data["sst"][:], dtype=np.float).reshape(-1, 36, 24, 72, 1)
#     t300 = np.array(train_data["t300"][:], dtype=np.float).reshape(-1, 36, 24, 72, 1)
#     ua = np.array(train_data["ua"][:], dtype=np.float).reshape(-1, 36, 24, 72, 1)
#     va = np.array(train_data["va"][:], dtype=np.float).reshape(-1, 36, 24, 72, 1)
#     features = np.concatenate([sst, t300, ua, va], axis=-1)
#
#     all_labels = labels[0].copy()
#     all_features = features[0].copy()
#     for i in range(1, labels.shape[0]):
#         all_labels = np.concatenate([all_labels, labels[i][24:]])
#         all_features = np.concatenate([all_features, features[i][24:]])
#
#     first_features, second_features, final_labels = list(), list(), list()
#     for i in range(all_labels.shape[0] - 36):
#         first_features.append(all_features[i:i + 12].tolist())
#         second_features.append(all_features[i + split:i + 6].tolist())
#         final_labels.append(labels[i + 12:i + 36].tolist())
#     print(first_features)
#     exit(0)
#
#     second_features = np.concatenate([sst[:, split:12, :, :].reshape(-1, 12 - split, 24, 72, 1),
#                                       t300[:, split:12, :, :].reshape(-1, 12 - split, 24, 72, 1),
#                                       ua[:, split:12, :, :].reshape(-1, 12 - split, 24, 72, 1),
#                                       va[:, split:12, :, :].reshape(-1, 12 - split, 24, 72, 1)], axis=-1)
#
#     labels = labels[:, 12:]
#
#     return first_features, second_features, labels
