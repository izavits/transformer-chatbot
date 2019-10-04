"""Dataloader unit tests"""

import unittest
from src.dataloader import preprocess, load_data, build_tokenizer, tokenize, construct_input


class TestDataLoader(unittest.TestCase):
    """Check the integrity of the data loader"""

    def setUp(self):
        self.datafile = 'data/SAMPLE.txt'

    def test_preprocess(self):
        """Check the preprocess line function"""
        line = 'I need help to set up my alarm, please!'
        output = preprocess(line)
        self.assertEqual(output, 'i need help to set up my alarm , please !')

    def test_load_data(self):
        """Check the load data function"""
        ins, outs = load_data(self.datafile)
        self.assertEqual(ins[5], 'i need help with my alarm')
        self.assertEqual(outs[5], 'sure , what would you like me to do ?')

    def test_build_tokenizer(self):
        """Check the build tokenizer function"""
        inputs, outputs = load_data(self.datafile)
        data_tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE = build_tokenizer(inputs, outputs)
        self.assertListEqual(data_tokenizer.encode(inputs[5]),
                             [1, 22, 23, 42, 14, 21])
        self.assertListEqual(data_tokenizer.encode(outputs[5]),
                             [66, 10, 12, 41, 5, 56, 30, 2, 108, 4])
        self.assertListEqual(START_TOKEN, [476])
        self.assertListEqual(END_TOKEN, [477])
        self.assertEqual(VOCAB_SIZE, 478)

    def test_tokenize(self):
        """Check the tokenize function"""
        inputs, outputs = load_data(self.datafile)
        questions, answers = tokenize(inputs, outputs)
        self.assertEqual(len(questions), 50)
        self.assertEqual(len(answers), 50)


suite = unittest.TestLoader().loadTestsFromTestCase(TestDataLoader)
unittest.TextTestRunner(verbosity=2).run(suite)
