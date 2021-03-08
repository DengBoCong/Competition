import tensorflow as tf
from code.layers import attention_layer
from code.layers import data_embedding
from code.layers import scaled_dot_product_attention
from code.tools import create_padding_mask
from typing import Any
from typing import AnyStr


def conv_layer(d_model: Any,
               d_type: tf.dtypes.DType = tf.float32, name: AnyStr = "conv_layer") -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None, None), dtype=d_type, name="{}_inputs")
    conv = tf.keras.layers.Conv1D(filters=d_model, kernel_size=3, padding="same", activation="relu")(inputs)
    outputs = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding="same")(conv)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def encoder_layer(batch_size: Any, d_model: Any, num_heads: Any, dropout: Any, d_type: tf.dtypes.DType = tf.float32,
                  name="encoder_layer") -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None, d_model), dtype=d_type, name="{}_inputs".format(name))
    padding_mask = tf.keras.Input(shape=(1, 1, None), dtype=d_type, name="{}_padding_mask".format(name))

    attention = scaled_dot_product_attention(num_heads=num_heads, depth=d_model // num_heads,
                                             d_type=d_type, mask=padding_mask)
    attention_output = attention_layer(batch_size=batch_size, d_model=d_model, num_heads=num_heads,
                                       attention=attention, d_type=d_type)(inputs=[inputs, inputs, inputs])

    attention_output = tf.keras.layers.Dropout(rate=dropout, dtype=d_type,
                                               name="{}_attention_dropout".format(name))(attention_output)
    attention_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype=d_type, name="{}_attention_layer_norm".format(name))(inputs + attention_output)

    conv_output = tf.keras.layers.Conv1D(filters=4 * d_model, kernel_size=1,
                                         strides=1, activation="relu")(attention_output)
    conv_output = tf.keras.layers.Dropout(rate=dropout, dtype=d_type,
                                          name="{}_outputs_dropout".format(name))(conv_output)
    conv_output = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1, strides=1)(conv_output)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype=d_type,
                                                 name="{}_outputs_layer_norm".format(name))(conv_output)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(num_layers: int, batch_size: int, d_model: int, num_heads: int,
            dropout: float, d_type: tf.dtypes.DType = tf.float32, name: str = "encoder") -> tf.keras.Model:
    """transformer的encoder

    :param num_layers: 编码解码的数量
    :param batch_size: batch大小
    :param d_model: 词嵌入维度
    :param num_heads: 多头注意力的头部层数量
    :param dropout: dropout的权重
    :param d_type: 运算精度
    :param name: 名称
    :return: Transformer的Encoder
    """
    inputs = tf.keras.Input(shape=(None, d_model), name="{}_inputs".format(name), dtype=d_type)
    padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),
                                          name="{}_padding_mask".format(name))(inputs)

    outputs = tf.keras.layers.Dropout(rate=dropout, dtype=d_type, name="{}_dropout".format(name))(inputs)

    for i in range(num_layers):
        enc_outputs = encoder_layer(batch_size=batch_size, d_model=d_model, num_heads=num_heads, dropout=dropout,
                                    d_type=d_type, name="{}_enc_layer_{}".format(name, i))([outputs, padding_mask])
        outputs = conv_layer(d_model=d_model, d_type=d_type, name="{}_conv_layer_{}".format(name, i))(enc_outputs)

    return tf.keras.Model(inputs=inputs, outputs=[outputs, padding_mask], name=name)


def informer(embedding_dim: Any,
             d_type: tf.dtypes.DType = tf.float32, name: AnyStr = "informer") -> tf.keras.Model:
    enc_inputs = tf.keras.Input(shape=(None, None), dtype=d_type, name="{}_enc_inputs".format(name))
    dec_inputs = tf.keras.Input(shape=(None, None), dtype=d_type, name="{}_dec_inputs".format(name))
    enc_month_inputs = tf.keras.Input(shape=(None,), dtype=d_type, name="{}_month_inputs".format(name))
    dec_month_inputs = tf.keras.Input(shape=(None,), dtype=d_type, name="{}_month_inputs".format(name))

    enc_embeddings = data_embedding(embedding_dim=embedding_dim, d_type=d_type)(inputs=[enc_inputs, enc_month_inputs])
    dec_embeddings = data_embedding(embedding_dim=embedding_dim, d_type=d_type)(inputs=[dec_inputs, dec_month_inputs])
