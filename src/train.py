"""Provides the functionality for training the model.
"""

import configparser
import tensorflow as tf
from model import transformer
from dataloader import load_data, build_tokenizer, tokenize, construct_input

__author__ = "ilias Zavitsanos"
__version__ = "1.0"
__maintainer__ = "ilias Zavitsanos"
__email__ = "izavits@gmail.com"
__status__ = "Research Ready"


class LearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Set up the learning rate. Use the Adam optimizer and the learning
    rate scheduler descibed in https://arxiv.org/abs/1706.03762"""

    def __init__(self, d_model, warmup_steps=4000):
        """Class constructor"""
        super(LearningRate, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Make it callable"""
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def main():
    """Main function"""
    config = configparser.ConfigParser()
    config.read('../config.ini')
    seed = int(config['MODEL']['TfRandomSeed'])
    tf.random.set_seed(seed)
    # Hyper-parameters
    NUM_LAYERS = int(config['MODEL']['NumLayers'])
    D_MODEL = int(config['MODEL']['Dmodel'])
    NUM_HEADS = int(config['MODEL']['NumHeads'])
    UNITS = int(config['MODEL']['Units'])
    DROPOUT = float(config['MODEL']['Dropout'])
    MAX_LENGTH = int(config['MODEL']['MaxLength'])
    EPOCHS = int(config['MODEL']['Epochs'])

    inputs, outputs = load_data()
    data_tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE = build_tokenizer(inputs, outputs)
    questions, answers = tokenize(inputs, outputs)
    dataset = construct_input(questions, answers)

    # Train the model
    tf.keras.backend.clear_session()
    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)

    def loss_function(y_true, y_pred):
        """Calculate the loss."""
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)
        return tf.reduce_mean(loss)

    def accuracy(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    # Set up learning rate and optimizer before compiling and fitting the model
    learning_rate = LearningRate(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
    model.fit(dataset, epochs=EPOCHS)
    return model


if __name__ == '__main__':
    main()
