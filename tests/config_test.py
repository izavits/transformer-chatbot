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

    def test_model_section_maxsamples(self):
        """Check that the model section has the MaxSamples parameter"""
        error_msg = 'MODEL section must have the MaxSamples parameter'
        model_section = self.config['MODEL']
        self.assertTrue('MaxSamples' in model_section, error_msg)

    def test_model_section_epochs(self):
        """Check that the model section has the Expochs parameter"""
        error_msg = 'MODEL section must have the Expochs parameter'
        model_section = self.config['MODEL']
        self.assertTrue('Epochs' in model_section, error_msg)

    def test_model_section_tfrandomseed(self):
        """Check that the model section has the TfRandomSeed parameter"""
        error_msg = 'MODEL section must have the TfRandomSeed parameter'
        model_section = self.config['MODEL']
        self.assertTrue('TfRandomSeed' in model_section, error_msg)

    def test_model_section_batchsize(self):
        """Check that the model section has the BatchSize parameter"""
        error_msg = 'MODEL section must have the BatchSize parameter'
        model_section = self.config['MODEL']
        self.assertTrue('BatchSize' in model_section, error_msg)

    def test_model_section_buffersize(self):
        """Check that the model section has the BufferSize parameter"""
        error_msg = 'MODEL section must have the BufferSize parameter'
        model_section = self.config['MODEL']
        self.assertTrue('BufferSize' in model_section, error_msg)

    def test_model_section_numlayers(self):
        """Check that the model section has the NumLayers parameter"""
        error_msg = 'MODEL section must have the NumLayers parameter'
        model_section = self.config['MODEL']
        self.assertTrue('NumLayers' in model_section, error_msg)

    def test_model_section_dmodel(self):
        """Check that the model section has the Dmodel parameter"""
        error_msg = 'MODEL section must have the Dmodel parameter'
        model_section = self.config['MODEL']
        self.assertTrue('Dmodel' in model_section, error_msg)

    def rest_model_section_numheads(self):
        """Check that the model section has the NumHeads parameter"""
        error_msg = 'MODEL section must have the NumHeads parameter'
        model_section = self.config['MODEL']
        self.assertTrue('NumHeads' in model_section, error_msg)

    def test_model_section_units(self):
        """Check that the model section has the Units parameter"""
        error_msg = 'MODEL section must have the Units parameter'
        model_section = self.config['MODEL']
        self.assertTrue('Units' in model_section, error_msg)

    def test_model_section_dropout(self):
        """Check that the model section has the Dropout parameter"""
        error_msg = 'MODEL section must have the Units parameter'
        model_section = self.config['MODEL']
        self.assertTrue('Dropout' in model_section, error_msg)

    def test_model_section_maxlength(self):
        """Check that the model section has the MaxLength parameter"""
        error_msg = 'MODEL section must have the MaxLength parameter'
        model_section = self.config['MODEL']
        self.assertTrue('MaxLength' in model_section, error_msg)


suite = unittest.TestLoader().loadTestsFromTestCase(TestConfigIntegrity)
unittest.TextTestRunner(verbosity=2).run(suite)
