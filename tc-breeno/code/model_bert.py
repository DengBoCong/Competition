from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from code.layers import MultiHeadAttention
from code.layers import scaled_dot_product_attention
from code.positional_encoding import positional_encoding
from code.tools import create_padding_mask


def pre_net(vocab_size: int, embedding_dim: int, dropout: float,
            d_type: tf.dtypes.DType = tf.float32, name: str = "preNet") -> tf.keras.Model:
    """ PreNet

    :param vocab_size: token大小
    :param embedding_dim: 词嵌入维度
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return: PreNet
    """
    query_inputs = tf.keras.Input(shape=(None,), name="{}_query_inputs".format(name))
    embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                           dtype=d_type, name="{}_embeddings".format(name))(query_inputs)

    embeddings *= tf.math.sqrt(x=tf.cast(x=embedding_dim, dtype=d_type), name="{}_sqrt".format(name))
    pos_encoding = positional_encoding(position=vocab_size, d_model=embedding_dim, d_type=d_type)
    embeddings = embeddings + pos_encoding[:, :tf.shape(embeddings)[1], :]

    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_dropout".format(name))(embeddings)
    return tf.keras.Model(inputs=query_inputs, outputs=outputs, name=name)


def block_net(units: int, d_model: int, num_heads: int, dropout: float,
              d_type: tf.dtypes.DType = tf.float32, name: str = "block_net") -> tf.keras.Model:
    """ BlockNet

    :param units: 词汇量大小
    :param d_model: 深度，词嵌入维度
    :param num_heads: 注意力头数
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return: BlockNet
    """
    query = tf.keras.Input(shape=(None, d_model), dtype=d_type, name="{}_query".format(name))
    padding_mask = tf.keras.Input(shape=(1, 1, None), dtype=d_type, name="{}_padding_mask".format(name))

    attention, _ = MultiHeadAttention(d_model, num_heads)(q=query, k=query, v=query, mask=padding_mask)
    attention = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_attention_dropout".format(name))(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                   name="{}_attention_layer_norm".format(name))(query + attention)

    outputs = tf.keras.layers.Dense(units=units, activation="gelu",
                                    dtype=d_type, name="{}_dense_act".format(name))(attention)
    outputs = tf.keras.layers.Dense(units=d_model, dtype=d_type, name="{}_dense".format(name))(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_outputs_dropout".format(name))(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                 name="{}_outputs_layer_norm".format(name))(attention + outputs)

    return tf.keras.Model(inputs=[query, padding_mask], outputs=outputs, name=name)


def match_net(vocab_size: int, num_layers: int, units: int, embedding_dim: int, num_heads: int, dropout: float,
              d_type: tf.dtypes.DType = tf.float32, name: str = "match_net") -> tf.keras.Model:
    """ MatchNet

    :param vocab_size: 词汇量大小
    :param num_layers: 编码解码的数量
    :param units: 单元大小
    :param embedding_dim: 词嵌入维度
    :param num_heads: 多头注意力的头部层数量
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return: MatchNet
    """
    query_inputs = tf.keras.Input(shape=(None,), name="{}_query_inputs".format(name))

    query_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),
                                                name="{}_first_padding_mask".format(name))(query_inputs)

    outputs = pre_net(vocab_size=vocab_size, embedding_dim=embedding_dim,
                      dropout=dropout, d_type=d_type, name="first_pre_net")(query_inputs)

    for i in range(num_layers):
        outputs = block_net(
            units=units, d_model=embedding_dim, num_heads=num_heads,
            dropout=dropout, d_type=d_type, name="{}_block_{}".format(name, i)
        )([outputs, query_padding_mask])

    return tf.keras.Model(inputs=query_inputs, outputs=outputs, name=name)


def last_net(embedding_dim: int, first_kernel_size: int = 3, second_kernel_size: int = 3, first_output_dim: int = 32,
             second_output_dim: int = 16, first_strides_size: int = 3, second_strides_size: int = 3,
             d_type: tf.dtypes.DType = tf.float32, name: str = "last_net") -> tf.keras.Model:
    """ LastNet

    :param embedding_dim: 词嵌入维度
    :param first_kernel_size: 第一个卷积核大小
    :param second_kernel_size: 第二个卷积核大小
    :param first_output_dim: 第一个卷积输出通道数
    :param second_output_dim: 第二个卷积输出通道数
    :param first_strides_size: 第一个卷积步幅大小
    :param second_strides_size: 第二个卷积步幅大小
    :param d_type: 运算精度
    :param name: 名称
    :return: LastNet
    """
    aggregation_inputs = tf.keras.Input(shape=(None, embedding_dim),
                                        dtype=d_type, name="{}_inputs".format(name))

    first_cnn_outputs = tf.keras.layers.Conv1D(
        filters=first_output_dim, kernel_size=first_kernel_size, activation="gelu",
        strides=first_strides_size, padding="same", name="{}_first_cnn".format(name)
    )(aggregation_inputs)

    first_max_pooling = tf.keras.layers.MaxPool1D(
        pool_size=first_kernel_size, strides=first_strides_size, padding="same",
        name="{}_first_max_pooling".format(name)
    )(first_cnn_outputs)

    second_cnn_outputs = tf.keras.layers.Conv1D(
        filters=second_output_dim, kernel_size=second_kernel_size, activation="gelu",
        strides=second_strides_size, padding="same", name="{}_second_cnn".format(name)
    )(first_max_pooling)

    second_max_pooling = tf.keras.layers.MaxPool1D(
        pool_size=second_kernel_size, strides=second_strides_size, padding="same",
        name="{}_second_max_pooling".format(name)
    )(second_cnn_outputs)

    outputs = tf.keras.layers.Flatten(dtype=d_type, name="{}_flatten".format(name))(second_max_pooling)

    return tf.keras.Model(inputs=aggregation_inputs, outputs=outputs, name=name)


def model(vocab_size: int, num_layers: int, units: int, embedding_dim: int, num_heads: int, dropout: float,
          max_sentence: int, first_kernel_size: int = 3, second_kernel_size: int = 3, first_strides_size: int = 3,
          second_strides_size: int = 3, first_output_dim: int = 32, second_output_dim: int = 16,
          d_type: tf.dtypes.DType = tf.float32, name: str = "model") -> tf.keras.Model:
    """ 核心模型

    :param vocab_size: 词汇量大小
    :param num_layers: 编码解码的数量
    :param units: 单元大小
    :param embedding_dim: 词嵌入维度
    :param num_heads: 多头注意力的头部层数量
    :param dropout: dropout的权重
    :param max_sentence: 最大句子长度
    :param first_kernel_size: 第一个卷积核大小
    :param second_kernel_size: 第二个卷积核大小
    :param first_strides_size: 第一个卷积步幅大小
    :param second_strides_size: 第二个卷积步幅大小
    :param first_output_dim: 第一个卷积输出通道数
    :param second_output_dim: 第二个卷积输出通道数
    :param d_type: 运算精度
    :param name: 名称
    :return: Model
    """
    query_input = tf.keras.Input(shape=(max_sentence,), name="{}_query_input".format(name))

    match_net_outputs = match_net(
        vocab_size=vocab_size, num_layers=num_layers, units=units,
        embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout, d_type=d_type
    )(query_input)

    # last_net_outputs = last_net(
    #     first_kernel_size=first_kernel_size, second_kernel_size=second_kernel_size,
    #     first_output_dim=first_output_dim, second_output_dim=second_output_dim, embedding_dim=embedding_dim,
    #     first_strides_size=first_strides_size, second_strides_size=second_strides_size, d_type=d_type
    # )(match_net_outputs)

    outputs = tf.keras.layers.Dense(units=vocab_size, activation="gelu", kernel_initializer="glorot_normal",
                                    dtype=d_type, name="{}_dense".format(name))(match_net_outputs)
    outputs = tf.keras.layers.Softmax(axis=-1, dtype=d_type, name="{}_softmax".format(name))(outputs)

    return tf.keras.Model(inputs=query_input, outputs=outputs, name=name)
