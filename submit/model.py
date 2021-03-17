import tensorflow as tf
from layers import attention_layer
from layers import data_embedding
from typing import Tuple
from typing import Any
from typing import AnyStr
from typing import TextIO
from typing import Dict


def conv_layer(d_model: Any, d_type: tf.dtypes.DType = tf.float32, name: AnyStr = "conv_layer") -> tf.keras.Model:
    """

    :param d_model:
    :param d_type:
    :param name:
    :return:
    """

    inputs = tf.keras.Input(shape=(12, d_model), dtype=d_type, name="{}_inputs".format(name))
    outputs = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding="same", activation="relu")(inputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def encoder_layer(batch_size: Any, d_model: Any, num_heads: Any, dropout: Any,
                  d_type: tf.dtypes.DType = tf.float32, name="encoder_layer") -> tf.keras.Model:
    """

    :param batch_size:
    :param d_model:
    :param num_heads:
    :param dropout:
    :param d_type:
    :param name:
    :return:
    """
    inputs = tf.keras.Input(shape=(12, d_model), dtype=d_type, name="{}_inputs".format(name))

    # attention = scaled_dot_product_attention(num_heads=num_heads, depth=d_model // num_heads,
    #                                          d_type=d_type, mask=padding_mask)
    attention_output = attention_layer(batch_size=batch_size, d_model=d_model, num_heads=num_heads,
                                       d_type=d_type)(inputs=[inputs, inputs, inputs])

    attention_output = tf.keras.layers.Dropout(rate=dropout, dtype=d_type,
                                               name="{}_attention_dropout".format(name))(attention_output)
    attention_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype=d_type, name="{}_attention_layer_norm".format(name))(inputs + attention_output)

    conv_output = tf.keras.layers.Conv1D(filters=4 * d_model, kernel_size=1,
                                         strides=1, activation="relu", padding="same")(attention_output)
    conv_output = tf.keras.layers.Dropout(rate=dropout, dtype=d_type,
                                          name="{}_outputs_dropout".format(name))(conv_output)
    conv_output = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1, strides=1, padding="same")(conv_output)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                 name="{}_outputs_layer_norm".format(name))(conv_output)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def encoder(num_layers: int, batch_size: int, embedding_dim: int, num_heads: int,
            dropout: float, d_type: tf.dtypes.DType = tf.float32, name: str = "encoder") -> tf.keras.Model:
    """transformer的encoder

    :param num_layers: 编码解码的数量
    :param batch_size: batch大小
    :param embedding_dim: 词嵌入维度
    :param num_heads: 多头注意力的头部层数量
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return: Transformer的Encoder
    """
    inputs = tf.keras.Input(shape=(12, embedding_dim), name="{}_inputs".format(name), dtype=d_type)

    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_dropout".format(name))(inputs)

    for i in range(num_layers):
        enc_outputs = encoder_layer(batch_size=batch_size, d_model=embedding_dim, num_heads=num_heads, dropout=dropout,
                                    d_type=d_type, name="{}_enc_layer_{}".format(name, i))(outputs)
        outputs = conv_layer(d_model=embedding_dim, d_type=d_type, name="{}_conv_layer_{}".format(name, i))(enc_outputs)

    outputs = encoder_layer(batch_size=batch_size, d_model=embedding_dim, num_heads=num_heads, dropout=dropout,
                            d_type=d_type, name="{}_enc_layer_{}".format(name, num_layers))(outputs)

    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                 name="{}_outputs_layer_norm".format(name))(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def decoder_layer(batch_size: Any, d_model: int, num_heads: int, dropout: float,
                  d_type: tf.dtypes.DType = tf.float32, name: str = "decoder_layer") -> tf.keras.Model:
    """Transformer的decoder层

    :param batch_size: batch大小
    :param units: 词汇量大小
    :param d_model: 深度，词嵌入维度
    :param num_heads: 注意力头数
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return: Transformer的Decoder内部层
    """
    inputs = tf.keras.Input(shape=(30, d_model), dtype=d_type, name="{}_inputs".format(name))
    enc_outputs = tf.keras.Input(shape=(12, d_model), dtype=d_type, name="{}_encoder_outputs".format(name))
    # look_ahead_mask = tf.keras.Input(shape=(1, None, None), dtype=d_type, name="{}_look_ahead_mask".format(name))

    # self_attention = scaled_dot_product_attention(num_heads=num_heads, depth=d_model // num_heads,
    #                                               d_type=d_type, mask=look_ahead_mask)
    self_attention_output = attention_layer(
        batch_size=batch_size, d_model=d_model, num_heads=num_heads,
        d_type=d_type, name="{}_attention_layer_1".format(name)
    )(inputs=[inputs, inputs, inputs])

    self_attention_output = tf.keras.layers.Dropout(rate=dropout, dtype=d_type,
                                                    name="{}_attention_dropout1".format(name))(self_attention_output)
    self_attention_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype=d_type, name="{}_attention_layer_norm1".format(name))(inputs + self_attention_output)

    # cross_attention = scaled_dot_product_attention(num_heads=num_heads, depth=d_model // num_heads,
    #                                                d_type=d_type, mask=padding_mask)
    cross_attention_output = attention_layer(
        batch_size=batch_size, d_model=d_model, num_heads=num_heads,
        d_type=d_type, name="{}_attention_layer_2".format(name)
    )(inputs=[self_attention_output, enc_outputs, enc_outputs])

    cross_attention_output = tf.keras.layers.Dropout(
        rate=dropout, dtype=d_type, name="{}_attention_dropout2".format(name))(cross_attention_output)
    cross_attention_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype=d_type,
        name="{}_attention_layer_norm2".format(name)
    )(self_attention_output + cross_attention_output)

    outputs = tf.keras.layers.Conv1D(filters=4 * d_model, kernel_size=1,
                                     strides=1, activation="relu")(cross_attention_output)
    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type,
                                      name="{}_outputs_dropout".format(name))(outputs)
    outputs = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1, strides=1)(outputs)

    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype=d_type, name="{}_outputs_layer_norm".format(name))(cross_attention_output + outputs)

    return tf.keras.Model(inputs=[inputs, enc_outputs], outputs=outputs, name=name)


def decoder(batch_size: Any, num_layers: int, embedding_dim: int, num_heads: int,
            dropout: float, d_type: tf.dtypes.DType = tf.float32, name: str = "decoder") -> tf.keras.Model:
    """transformer的decoder

    :param batch_size: batch大小
    :param num_layers: 编码解码的数量
    :param embedding_dim: 词嵌入维度
    :param num_heads: 多头注意力的头部层数量
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return: Transformer的Decoder
    """
    inputs = tf.keras.Input(shape=(30, embedding_dim), dtype=d_type, name="{}_inputs".format(name))
    enc_outputs = tf.keras.Input(shape=(12, embedding_dim), dtype=d_type, name="{}_encoder_outputs".format(name))

    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_dropout".format(name))(inputs)

    for i in range(num_layers):
        outputs = decoder_layer(
            batch_size=batch_size, d_model=embedding_dim, num_heads=num_heads, dropout=dropout,
            d_type=d_type, name="decoder_layer_{}".format(i))(inputs=[outputs, enc_outputs])

    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                 name="{}_outputs_layer_norm".format(name))(outputs)

    return tf.keras.Model(inputs=[inputs, enc_outputs], outputs=outputs, name=name)


def informer(embedding_dim: Any, enc_num_layers: Any, dec_num_layers: Any, batch_size: Any, num_heads: Any, dropout: Any,
             d_type: tf.dtypes.DType = tf.float32, name: AnyStr = "informer") -> tf.keras.Model:
    enc_inputs = tf.keras.Input(shape=(24, 72, 4), dtype=d_type, name="{}_enc_inputs".format(name))
    dec_inputs = tf.keras.Input(shape=(24, 72, 4), dtype=d_type, name="{}_dec_inputs".format(name))
    enc_month_inputs = tf.keras.Input(shape=(12,), dtype=d_type, name="{}_enc_month_inputs".format(name))
    dec_month_inputs = tf.keras.Input(shape=(30,), dtype=d_type, name="{}_dec_month_inputs".format(name))

    enc_feature = tf.keras.layers.Conv2D(filters=1, kernel_size=3)(enc_inputs)
    enc_feature = tf.keras.layers.Flatten()(enc_feature)
    enc_feature = tf.expand_dims(input=tf.keras.layers.Dense(units=embedding_dim)(enc_feature), axis=0)
    dec_feature = tf.keras.layers.Conv2D(filters=1, kernel_size=3)(dec_inputs)
    dec_feature = tf.keras.layers.Flatten()(dec_feature)
    dec_feature = tf.expand_dims(input=tf.keras.layers.Dense(units=embedding_dim)(dec_feature), axis=0)

    enc_embeddings = data_embedding(embedding_dim=embedding_dim, d_type=d_type,
                                    name="data_embedding_1")(inputs=[enc_feature, enc_month_inputs])
    enc_outputs = encoder(num_layers=enc_num_layers, batch_size=batch_size, embedding_dim=embedding_dim,
                          num_heads=num_heads, dropout=dropout, d_type=d_type)(enc_embeddings)
    dec_embeddings = data_embedding(embedding_dim=embedding_dim, d_type=d_type,
                                    name="data_embedding_2")(inputs=[dec_feature, dec_month_inputs])
    dec_outputs = decoder(batch_size=batch_size, num_layers=dec_num_layers, embedding_dim=embedding_dim,
                          num_heads=num_heads, dropout=dropout, d_type=d_type)(inputs=[dec_embeddings, enc_outputs])

    outputs = tf.keras.layers.Dense(units=1)(dec_outputs)

    return tf.keras.Model(inputs=[enc_inputs, dec_inputs, enc_month_inputs, dec_month_inputs],
                          outputs=outputs, name=name)
