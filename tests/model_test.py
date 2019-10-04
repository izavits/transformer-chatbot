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


suite = unittest.TestLoader().loadTestsFromTestCase(TestModel)
unittest.TextTestRunner(verbosity=2).run(suite)
