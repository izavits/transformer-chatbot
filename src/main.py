"""Provides the functionality of the inference.
Evaluates the input and provides the decoded output.
"""

import configparser
import argparse
from argparse import RawTextHelpFormatter
import tensorflow as tf
from dataloader import preprocess, load_data, build_tokenizer
from model import transformer
from train import main

__author__ = "ilias Zavitsanos"
__version__ = "1.0"
__maintainer__ = "ilias Zavitsanos"
__email__ = "izavits@gmail.com"
__status__ = "Research Ready"


def load_model():
    """Load the model, the weights and the needed parameters"""
    config = configparser.ConfigParser()
    config.read('../config.ini')
    model_name = config['DATA']['InputSet'].split('/')[-1].split('.')[0]
    with open('../models/' + model_name + '_config.json') as json_file:
        json_config = json_file.read()
    # Hyper-parameters
    NUM_LAYERS = int(config['MODEL']['NumLayers'])
    D_MODEL = int(config['MODEL']['Dmodel'])
    NUM_HEADS = int(config['MODEL']['NumHeads'])
    UNITS = int(config['MODEL']['Units'])
    DROPOUT = float(config['MODEL']['Dropout'])
    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)
    model.load_weights('../models/' + model_name + '_model.h5')
    return model


def get_extreme_tokens():
    """Get start and end tokens"""
    config = configparser.ConfigParser()
    config.read('../config.ini')
    datafile = '../' + config['DATA']['InputSet']
    inputs, outputs = load_data(datafile)
    data_tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE = build_tokenizer(inputs, outputs)
    return START_TOKEN, END_TOKEN, data_tokenizer, VOCAB_SIZE


def evaluate(utterance):
    """Evaluate the given utterance and return output"""
    utterance = preprocess(utterance)
    utterance = tf.expand_dims(START_TOKEN + data_tokenizer.encode(utterance) + END_TOKEN, axis=0)
    output = tf.expand_dims(START_TOKEN, 0)
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[utterance, output], training=False)
        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break
        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)
    return tf.squeeze(output, axis=0)


def predict(utterance):
    prediction = evaluate(utterance)
    predicted_sentence = data_tokenizer.decode(
        [i for i in prediction if i < data_tokenizer.vocab_size])
    print('Input: {}'.format(utterance))
    print('Output: {}'.format(predicted_sentence))
    return predicted_sentence


if __name__ == '__main__':
    """Main method"""
    # Get necessary parameters
    START_TOKEN, END_TOKEN, data_tokenizer, VOCAB_SIZE = get_extreme_tokens()
    config = configparser.ConfigParser()
    config.read('../config.ini')
    MAX_LENGTH = int(config['MODEL']['MaxLength'])
    # Parse command line arguments
    welcome = "Welcome to chatbot.\n"
    welcome += "Use the config.ini file to setup the required parameters and input dataset."
    parser = argparse.ArgumentParser(description=welcome, formatter_class=RawTextHelpFormatter)

    parser.add_argument('--train', '-t', required=False,
                        help='train the model before using for chatting',
                        action='store_true')
    parser.parse_args()
    args = parser.parse_args()
    train = args.train
    if train:
        # First train the model and then use it to make predictions
        main()
    model = load_model()
    # Make predictions
    output = predict('i need help with my alarm')
    output = predict('remove my alarm at 7 am')

