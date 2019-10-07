"""Model functions unit tests"""

import unittest
from src.model import PositionalEncoding, create_padding_mask, create_look_ahead_mask
import tensorflow as tf


class TestModel(unittest.TestCase):
    """Check the integrity of model functions"""

    def setUp(self):
        self.fake_data = [
            [1, 2, 3, 4, 5]
        ]

    def test_positional_encoding(self):
        """Check the positional encoding"""
        pe = PositionalEncoding(2, 2)
        self.assertEqual(str(pe.pos_encoding.numpy()[0]), '''[[0.         1.        ]
 [0.84147096 0.5403023 ]]''')

    def test_model_name_parsing(self):
        """Check the parsing of the model name"""
        datafile = './data/SAMPLE_SET.txt'
        model_name = datafile.split('/')[-1].split('.')[0]
        self.assertEqual(model_name, 'SAMPLE_SET')

    def test_padding_mask(self):
        """Check the padding mask"""
        mask = create_padding_mask(self.fake_data)
        self.assertEqual(mask.shape, tf.TensorShape([1, 1, 1, 5]))

    def test_look_ahead_mask(self):
        """Check the look ahead mask"""
        mask = create_look_ahead_mask(self.fake_data)
        self.assertEqual(mask.shape, tf.TensorShape([1, 1, 5, 5]))


suite = unittest.TestLoader().loadTestsFromTestCase(TestModel)
unittest.TextTestRunner(verbosity=2).run(suite)
