from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from code.layers import MultiHeadAttention
from code.positional_encoding import positional_encoding
from code.tools import create_padding_mask


def infer_sent(vocab_size: int, num_layers: int, units: int, embedding_dim: int, num_heads: int,
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

    premise_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),
                                                  name="{}_pre_padding_mask".format(name))(premise_inputs)
    hypothesis_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),
                                                     name="{}_hyp_padding_mask".format(name))(hypothesis_inputs)

    premise_embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                                   dtype=d_type, name="{}_pre_embeddings".format(name))(premise_inputs)
    hypothesis_embeddings = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_dim,
        dtype=d_type, name="{}_hyp_embeddings".format(name)
    )(premise_inputs)


    u, v = extract_net(
        vocab_size=vocab_size, num_layers=num_layers, units=units,
        embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout, d_type=d_type
    )(inputs=[premise_embeddings, hypothesis_embeddings, premise_padding_mask, hypothesis_padding_mask])

    outputs = feature_net(embedding_dim=embedding_dim, dropout=dropout, d_type=d_type)(inputs=[u, v])
    outputs = tf.nn.softmax(logits=outputs, axis=-1)

    return tf.keras.Model(inputs=[premise_inputs, hypothesis_inputs], outputs=outputs, name=name)


def feature_net(embedding_dim: int, dropout: float, d_type: tf.dtypes.DType = tf.float32, name: str = "model"):
    """ 特征处理层

    :param embedding_dim: 词嵌入维度
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return:
    """

    premise_inputs = tf.keras.Input(shape=(None, embedding_dim), name="{}_pre_inputs".format(name), dtype=d_type)
    hypothesis_inputs = tf.keras.Input(shape=(None, embedding_dim), name="{}_hyp_inputs".format(name), dtype=d_type)
    #
    # initializer = tf.random_normal_initializer(0.0, 0.1)

    u = tf.reduce_max(premise_inputs, axis=1)
    v = tf.reduce_max(hypothesis_inputs, axis=1)

    diff = tf.abs(tf.subtract(u, v))
    mul = tf.multiply(u, v)

    # features = tf.concat([u, v, diff, mul], axis=-1)
    features = tf.concat([diff, mul], axis=-1)

    features_drop = tf.keras.layers.Dropout(rate=dropout, dtype=d_type)(features)
    features_outputs = tf.keras.layers.Dense(units=embedding_dim, activation="relu")(features_drop)

    outputs_drop = tf.keras.layers.Dropout(rate=dropout, dtype=d_type)(features_outputs)
    outputs = tf.keras.layers.Dense(units=2, activation="relu")(outputs_drop)



    return tf.keras.Model(inputs=[premise_inputs, hypothesis_inputs], outputs=outputs, name=name)


def extract_net(vocab_size: int, num_layers: int, units: int, embedding_dim: int, num_heads: int,
                dropout: float, d_type: tf.dtypes.DType = tf.float32, name: str = "extract_net") -> tf.keras.Model:
    """ 特征抽取层

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
    premise_embeddings = tf.keras.Input(shape=(None, embedding_dim), name="{}_pre_inputs".format(name), dtype=d_type)
    hypothesis_embeddings = tf.keras.Input(shape=(None, embedding_dim), name="{}_hyp_inputs".format(name), dtype=d_type)
    premise_padding_mask = tf.keras.Input(shape=(1, 1, None), name="{}_pre_padding_mask".format(name), dtype=d_type)
    hypothesis_padding_mask = tf.keras.Input(shape=(1, 1, None), name="{}_hyp_padding_mask".format(name), dtype=d_type)

    u_outputs = encoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units, embedding_dim=embedding_dim,
        num_heads=num_heads, dropout=dropout, d_type=d_type, name="premise_encoder"
    )(inputs=[premise_embeddings, premise_padding_mask])
    v_outputs = encoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units, embedding_dim=embedding_dim,
        num_heads=num_heads, dropout=dropout, d_type=d_type, name="hypothesis_encoder"
    )(inputs=[hypothesis_embeddings, hypothesis_padding_mask])

    # u_premise_lstm = bi_lstm_block(hidden_size=embedding_dim // 2, embedding_dim=embedding_dim,
    #                                d_type=d_type, name="{}_pre_bi_lstm".format(name))(premise_embeddings)
    # v_hypothesis_lstm = bi_lstm_block(hidden_size=embedding_dim // 2, embedding_dim=embedding_dim,
    #                                   d_type=d_type, name="{}_hyp_bi_lstm".format(name))(hypothesis_embeddings)

    # u_outputs = tf.concat([u_premise, u_premise_lstm], axis=-1)
    # v_outputs = tf.concat([v_hypothesis, v_hypothesis_lstm], axis=-1)

    # u_outputs = u_premise + u_premise_lstm
    # v_outputs = v_hypothesis + v_hypothesis_lstm
    # u_outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type)(u_premise + u_premise_lstm)
    # v_outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type)(v_hypothesis + v_hypothesis_lstm)
    #
    # u_outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type)(u_premise)
    # v_outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type)(v_hypothesis)
    #


    return tf.keras.Model(inputs=[premise_embeddings, hypothesis_embeddings, premise_padding_mask,
                                  hypothesis_padding_mask], outputs=[u_outputs, v_outputs], name=name)


def bi_lstm_block(hidden_size: int, embedding_dim: int,
                  d_type: tf.dtypes.DType = tf.float32, name: str = "bi_lstm") -> tf.keras.Model:
    """ 双向LSTM

    :param hidden_size: 单元大小
    :param embedding_dim: 词嵌入维度
    :param d_type: 运算精度
    :param name: 名称
    :return:
    """
    inputs = tf.keras.Input(shape=(None, embedding_dim), name="{}_inputs".format(name), dtype=d_type)

    lstm = tf.keras.layers.LSTM(
        units=hidden_size, return_sequences=True, return_state=True,
        recurrent_initializer="glorot_uniform", dtype=d_type, name="{}_lstm_cell".format(name)
    )
    bi_lstm = tf.keras.layers.Bidirectional(layer=lstm, dtype=d_type, name="{}_bi_lstm".format(name))

    outputs = bi_lstm(inputs)[0]

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


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
    inputs = tf.keras.Input(shape=(None, embedding_dim), name="{}_inputs".format(name), dtype=d_type)
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="{}_padding_mask".format(name), dtype=d_type)

    embeddings = inputs * tf.math.sqrt(x=tf.cast(x=embedding_dim, dtype=d_type), name="{}_sqrt".format(name))
    pos_encoding = positional_encoding(position=vocab_size, d_model=embedding_dim, d_type=d_type)
    embeddings = embeddings + pos_encoding[:, :tf.shape(embeddings)[1], :]

    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_dropout".format(name))(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(units=units, d_model=embedding_dim, num_heads=num_heads, dropout=dropout,
                                d_type=d_type, name="{}_layer_{}".format(name, i))([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


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
