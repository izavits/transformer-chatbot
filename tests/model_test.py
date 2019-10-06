"""Model functions unit tests"""

import unittest
from src.model import PositionalEncoding


class TestModel(unittest.TestCase):
    """Check the integrity of model functions"""

    def setUp(self) -> None:
        pass

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


suite = unittest.TestLoader().loadTestsFromTestCase(TestModel)
unittest.TextTestRunner(verbosity=2).run(suite)
