"""Provides utility functions to load the input dataset
and process it as needed for the model.
"""

import re
import configparser
import json
import tensorflow as tf
import tensorflow_datasets as tfds

__author__ = "ilias Zavitsanos"
__version__ = "1.0"
__maintainer__ = "ilias Zavitsanos"
__email__ = "izavits@gmail.com"
__status__ = "Research Ready"


def preprocess(line):
    """Preprocess the input line by removing special characters."""
    line = line.lower().strip()
    line = re.sub(r"([?.!,])", r" \1 ", line)
    line = re.sub(r'[" "]+', " ", line)
    line = re.sub(r"[^a-zA-Z?.!,]+", " ", line)
    line = line.strip()
    return line


def load_data(datafile):
    """Load the dataset"""
    inputs, outputs = [], []
    # Input data is not valid json to load at once
    with open(datafile) as f:
        lines = f.readlines()
    data = [json.loads(l) for l in lines]
    for d in data:
        inputs += [preprocess(i) for (x, i) in enumerate(d['turns']) if x % 2 != 0]
        outputs += [preprocess(i) for (x, i) in enumerate(d['turns']) if x % 2 == 0 and x != 0]
    return inputs, outputs


def build_tokenizer(ins, outs):
    """Build a topkenizer using Tensorflow's SubwordTextEncoder."""
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(ins + outs, target_vocab_size=2 ** 13)
    start_token = [tokenizer.vocab_size]
    end_token = [tokenizer.vocab_size + 1]
    vocabulary_size = tokenizer.vocab_size + 2
    return tokenizer, start_token, end_token, vocabulary_size


def tokenize(inputs, outputs):
    """Tokenize and pad data."""
    tokenized_ins, tokenized_outs = [], []
    data_tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE = build_tokenizer(inputs, outputs)
    for (question, answer) in zip(inputs, outputs):
        question = START_TOKEN + data_tokenizer.encode(question) + END_TOKEN
        answer = START_TOKEN + data_tokenizer.encode(answer) + END_TOKEN
        tokenized_ins.append(question)
        tokenized_outs.append(answer)
    tokenized_ins = tf.keras.preprocessing.sequence.pad_sequences(tokenized_ins, maxlen=80, padding='post')
    tokenized_outs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_outs, maxlen=80, padding='post')
    return tokenized_ins, tokenized_outs


def construct_input(questions, answers):
    """Construct the input for the model."""
    config = configparser.ConfigParser()
    config.read('../config.ini')
    BATCH_SIZE = int(config['MODEL']['BatchSize'])
    BUFFER_SIZE = int(config['MODEL']['BufferSize'])
    # Use the tensorflow data API to exploit caching and prefetching features
    # Use teacher - forcing: pass the true output to the next step
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions,
            'dec_inputs': answers[:, :-1]
        },
        {
            'outputs': answers[:, 1:]
        },
    ))
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
