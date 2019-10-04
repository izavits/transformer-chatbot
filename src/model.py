"""Provides the functionality of the model.
The model is a Transformer, a neural network architecture
based on self-attention mechanism.
"""

import tensorflow as tf

__author__ = "ilias Zavitsanos"
__version__ = "1.0"
__maintainer__ = "ilias Zavitsanos"
__email__ = "izavits@gmail.com"
__status__ = "Research Ready"


def attention_weights(query, key, value, mask):
    """Calculate attention weights based on the
    scaled dot-product attention function."""
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    if mask is not None:
        logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output


def create_padding_mask(x):
    """Mask pad tokens in order not to treat padding as input."""
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    """Mask future tokens in a sequence."""
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi head attention consists of:
    - Linear layers and split into heads
    - Scaled dot-product attention
    - Heads concatenation
    - Final linear layer"""

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        """Class constructor."""
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)
        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, ins, batch_size):
        """Split into heads"""
        ins = tf.reshape(
            ins, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(ins, perm=[0, 2, 1, 3])

    def __call__(self, ins):
        """Make it callable. Use scaled dot-product attention
        and concatenate heads."""
        query, key, value, mask = ins['query'], ins['key'], ins[
            'value'], ins['mask']
        batch_size = tf.shape(query)[0]
        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        # scaled dot-product attention
        scaled_attention = attention_weights(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        # final linear layer
        outs = self.dense(concat_attention)
        return outs


class PositionalEncoding(tf.keras.layers.Layer):
    """Add positional encoding to give more info about the
    relative position of the words in a sentence. The specific
    model by itself makes no assumptions about the spatial
    relationships across the data."""

    def __init__(self, position, d_model):
        """Class constructor"""
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        """Calculate anglr rads"""
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        """Perform positional encoding"""
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin and cos to even and odd indexes in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def __call__(self, inputs):
        """Make it callable"""
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    """The encoder layer consists of a multi head attention sublayer
    and two dense layers followed by dropout."""
    ins = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({'query': ins,
                                               'key': ins,
                                               'value': ins,
                                               'mask': padding_mask
                                               })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ins + attention)
    outs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outs = tf.keras.layers.Dense(units=d_model)(outs)
    outs = tf.keras.layers.Dropout(rate=dropout)(outs)
    outs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outs)
    return tf.keras.Model(inputs=[ins, padding_mask], outputs=outs, name=name)


def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder"):
    """The encoder consists of the input embedding, the positional encoding
    and the given number of layers."""
    ins = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(ins)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
    for i in range(num_layers):
        outs = encoder_layer(units=units,
                             d_model=d_model,
                             num_heads=num_heads,
                             dropout=dropout,
                             name="encoder_layer_{}".format(i),
                             )([outs, padding_mask])
    return tf.keras.Model(
        inputs=[ins, padding_mask], outputs=outs, name=name)


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    """The decoder layer consists of a masked multi head attention layer,
    a multi head attention layer that receives the encoder output and the
    output of the masked multi head attention sublayer, and two dense
    layers followed by dropout."""
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")({'query': inputs,
                                                 'key': inputs,
                                                 'value': inputs,
                                                 'mask': look_ahead_mask
                                                 })
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention1 + inputs)
    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention_2")({'query': attention1,
                                                 'key': enc_outputs,
                                                 'value': enc_outputs,
                                                 'mask': padding_mask
    })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2 + attention1)
    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs + attention2)
    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def decoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name='decoder'):
    """The decoder consists of output embedding, the positional encoding and the
    given number of decoder layers."""
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])
    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def transformer(vocab_size, num_layers, units, d_model, num_heads, dropout, name="transformer"):
    """The transformer consists of the encoder, decoder and a final linear layer.
    The output of the decoder constitutes the input to the final linear layer and
    its output is returned."""
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")
    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)
    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask,
        output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)
    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)
    enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs, enc_padding_mask])
    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)
    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
