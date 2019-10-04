"""Config unit tests"""

import unittest
import configparser


class TestConfigIntegrity(unittest.TestCase):
    """Check if the config file is complete"""

    def setUp(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

    def test_num_sections(self):
        """Check if the config has two sections"""
        sections = self.config.sections()
        self.assertEqual(len(sections), 2)

    def test_model_section(self):
        """Check that the model section exists"""
        error_msg = 'Config must have a MODEL section.'
        sections = self.config.sections()
        self.assertTrue('MODEL' in sections, error_msg)

    def test_data_section(self):
        """Check that the data section exists"""
        error_msg = 'Config must have a DATA section'
        sections = self.config.sections()
        self.assertTrue('DATA' in sections, error_msg)

    def test_data_input_set(self):
        """Check that the data section has input set"""
        error_msg = 'DATA section must have InputSet'
        data_section = self.config['DATA']
        self.assertTrue('InputSet' in data_section, error_msg)

    def test_model_section_params(self):
        """Check that the model section has all the needed parameters"""
        error_msg = 'MODEL section must have all needed parameters'
        model_section = self.config['MODEL']
        self.assertTrue('MaxSamples' in model_section, error_msg)
        self.assertTrue('Epochs' in model_section, error_msg)
        self.assertTrue('TfRandomSeed' in model_section, error_msg)
        self.assertTrue('BatchSize' in model_section, error_msg)
        self.assertTrue('BufferSize' in model_section, error_msg)
        self.assertTrue('NumLayers' in model_section, error_msg)
        self.assertTrue('Dmodel' in model_section, error_msg)
        self.assertTrue('NumHeads' in model_section, error_msg)
        self.assertTrue('Units' in model_section, error_msg)
        self.assertTrue('Dropout' in model_section, error_msg)
        self.assertTrue('MaxLength' in model_section, error_msg)


suite = unittest.TestLoader().loadTestsFromTestCase(TestConfigIntegrity)
unittest.TextTestRunner(verbosity=2).run(suite)
