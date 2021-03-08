import gc
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from typing import Any
from typing import AnyStr
from typing import Tuple


def preprocess_data_diff(train_data_path: AnyStr, label_data_path: AnyStr, save_dir: AnyStr,
                         data_type: AnyStr, split: Any = 6) -> Tuple:
    """ 预处理数据集

    :param train_data_path: 训练集所在路径
    :param label_data_path: 标签数据集所在路径
    :param save_dir: 保存目录
    :param data_type: 数据类型
    :param split: 对训练数据进行切分的起始位置
    :return: 无返回值
    """
    train_data = Dataset(filename=train_data_path, mode="r")
    label_data = Dataset(filename=label_data_path, mode="r")

    # labels = np.array(label_data["nino"][:])
    # sst = np.array(train_data["sst"][:])
    # t300 = np.array(train_data["t300"][:])
    # ua = np.array(train_data["ua"][:])
    # va = np.array(train_data["va"][:])
    flag = 100 if data_type == "soda" else 4645

    for i in range(flag):
        np.save(file=save_dir, arr=np.concatenate(
            [np.array(train_data["sst"][i]).reshape(-1, 24, 72, 1),
             np.array(train_data["t300"][i]).reshape(-1, 24, 72, 1),
             np.array(train_data["ua"][i]).reshape(-1, 24, 72, 1),
             np.array(train_data["va"][i]).reshape(-1, 24, 72, 1)], axis=-1
        )[:12, :, :, :])
        gc.collect()
        exit(0)

    first_features = np.concatenate(
        [sst[:, :12, :, :].reshape(-1, 12, 24, 72, 1), t300[:, :12, :, :].reshape(-1, 12, 24, 72, 1),
         ua[:, :12, :, :].reshape(-1, 12, 24, 72, 1), va[:, :12, :, :].reshape(-1, 12, 24, 72, 1)], axis=-1
    )

    second_features = np.concatenate([sst[:, split:12, :, :].reshape(-1, 12 - split, 24, 72, 1),
                                      t300[:, split:12, :, :].reshape(-1, 12 - split, 24, 72, 1),
                                      ua[:, split:12, :, :].reshape(-1, 12 - split, 24, 72, 1),
                                      va[:, split:12, :, :].reshape(-1, 12 - split, 24, 72, 1)], axis=-1)

    labels = labels[:, 12:]

    return first_features, second_features, labels

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
