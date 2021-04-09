from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from code.layers import MultiHeadAttention
from code.positional_encoding import positional_encoding
from code.tools import create_padding_mask


def model(vocab_size: int, num_layers: int, units: int, embedding_dim: int, num_heads: int,
          dropout: float, d_type: tf.dtypes.DType = tf.float32, name: str = "model") -> tf.keras.Model:
    """短文本匹配模型

    :param vocab_size: token大小
    :param num_layers: 编码解码的数量
    :param units: 单元大小
    :param embedding_dim: 词嵌入维度
    :param num_heads: 多头注意力的头部层数量
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return:
    """
    premise_inputs = tf.keras.Input(shape=(None,), name="{}_premise_inputs".format(name), dtype=d_type)
    hypothesis_inputs = tf.keras.Input(shape=(None,), name="{}_hypothesis_inputs".format(name), dtype=d_type)
    # initializer = tf.random_normal_initializer(0.0, 0.1)

    u_premise, premise_padding_mask = encoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units, embedding_dim=embedding_dim,
        num_heads=num_heads, dropout=dropout, d_type=d_type, name="premise_encoder")(premise_inputs)
    v_hypothesis, hypothesis_padding_mask = encoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units, embedding_dim=embedding_dim,
        num_heads=num_heads, dropout=dropout, d_type=d_type, name="hypothesis_encoder")(hypothesis_inputs)

    u = tf.reduce_max(u_premise, axis=1)
    v = tf.reduce_max(v_hypothesis, axis=1)

    diff = tf.abs(tf.subtract(u, v))
    mul = tf.multiply(u, v)

    features = tf.concat([u, v, diff, mul], axis=-1)
    # features = tf.concat([diff, mul], axis=-1)

    features_drop = tf.keras.layers.Dropout(rate=dropout, dtype=d_type)(features)
    features_outputs = tf.keras.layers.Dense(units=embedding_dim, activation="relu")(features_drop)

    outputs_drop = tf.keras.layers.Dropout(rate=dropout, dtype=d_type)(features_outputs)
    outputs = tf.keras.layers.Dense(units=2, activation="relu")(outputs_drop)
    outputs = tf.nn.softmax(logits=outputs, axis=-1)

    return tf.keras.Model(inputs=[premise_inputs, hypothesis_inputs], outputs=outputs, name=name)


def encoder(vocab_size: int, num_layers: int, units: int, embedding_dim: int, num_heads: int,
            dropout: float, d_type: tf.dtypes.DType = tf.float32, name: str = "encoder") -> tf.keras.Model:
    """ 文本句子编码

    :param vocab_size: token大小
    :param num_layers: 编码解码的数量
    :param units: 单元大小
    :param embedding_dim: 词嵌入维度
    :param num_heads: 多头注意力的头部层数量
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return:
    """
    inputs = tf.keras.Input(shape=(None,), name="{}_inputs".format(name), dtype=d_type)
    padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),
                                          name="{}_padding_mask".format(name))(inputs)
    embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                           dtype=d_type, name="{}_embeddings".format(name))(inputs)
    embeddings *= tf.math.sqrt(x=tf.cast(x=embedding_dim, dtype=d_type), name="{}_sqrt".format(name))
    pos_encoding = positional_encoding(position=vocab_size, d_model=embedding_dim, d_type=d_type)
    embeddings = embeddings + pos_encoding[:, :tf.shape(embeddings)[1], :]

    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_dropout".format(name))(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(units=units, d_model=embedding_dim, num_heads=num_heads, dropout=dropout,
                                d_type=d_type, name="{}_layer_{}".format(name, i))([outputs, padding_mask])

    return tf.keras.Model(inputs=inputs, outputs=[outputs, padding_mask], name=name)


def encoder_layer(units: int, d_model: int, num_heads: int, dropout: float,
                  d_type: tf.dtypes.DType = tf.float32, name: str = "encoder_layer") -> tf.keras.Model:
    """
    :param units: 词汇量大小
    :param d_model: 深度，词嵌入维度
    :param num_heads: 注意力头数
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return:
    """
    inputs = tf.keras.Input(shape=(None, d_model), dtype=d_type, name="{}_inputs".format(name))
    padding_mask = tf.keras.Input(shape=(1, 1, None), dtype=d_type, name="{}_padding_mask".format(name))

    attention, _ = MultiHeadAttention(d_model, num_heads)(q=inputs, k=inputs, v=inputs, mask=padding_mask)
    attention = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_attention_dropout".format(name))(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                   name="{}_attention_layer_norm".format(name))(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation="relu",
                                    dtype=d_type, name="{}_dense_act".format(name))(attention)
    outputs = tf.keras.layers.Dense(units=d_model, dtype=d_type, name="{}_dense".format(name))(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_outputs_dropout".format(name))(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                 name="{}_outputs_layer_norm".format(name))(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)
