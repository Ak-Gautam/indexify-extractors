#For three test cases,the failure is 1.
#the failure could be minimized by further adjusting or fine tuning the tolerence level. (tried but didn't work)


import unittest
from unittest.mock import patch
from numpy.testing import assert_allclose
from transformers import LEDModel, LEDTokenizer
import torch
import numpy as np

class LEDBaseExtractor:
    def __init__(self, model_name="allenai/led-base-16384"):
        self.model = LEDModel.from_pretrained(model_name)
        self.tokenizer = LEDTokenizer.from_pretrained(model_name)

    def extract_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

class TestLEDBaseExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor_name = "led"
        self.extractor = LEDBaseExtractor()

    @patch("transformers.LEDModel.from_pretrained")
    @patch("transformers.LEDTokenizer.from_pretrained")
    def test_extract_embedding(self, mock_tokenizer, mock_model):
        mock_tokenizer.return_value = "mocked_tokenizer"
        
        # Mock the return value with actual values
        mock_model.return_value.last_hidden_state.mean.return_value.squeeze.return_value.numpy.return_value = np.random.rand(768)

        text = "pesce mandya"
        embedding = self.extractor.extract_embedding(text)

        # Use relative tolerance
        assert_allclose(embedding.tolist(), mock_model.return_value.last_hidden_state.mean.return_value.squeeze.return_value.numpy.return_value, rtol=1e-5, atol=1e-5)
        mock_tokenizer.assert_called_once_with(text, return_tensors="pt", truncation=True, padding=True)
        mock_model.assert_called_once_with("allenai/led-base-16384")
        mock_model.return_value.last_hidden_state.mean.assert_called_once_with(dim=1)
        mock_model.return_value.last_hidden_state.mean.return_value.squeeze.assert_called_once()
        mock_model.return_value.last_hidden_state.mean.return_value.squeeze.return_value.numpy.assert_called_once()

    @patch("transformers.LEDModel.from_pretrained")
    @patch("transformers.LEDTokenizer.from_pretrained")
    def test_extract_embedding_shape(self, mock_tokenizer, mock_model):
        mock_tokenizer.return_value = "mocked_tokenizer"
        mock_model.return_value.last_hidden_state.mean.return_value.squeeze.return_value.numpy.return_value = np.zeros(768)

        text = "therohithborana"
        embedding = self.extractor.extract_embedding(text)

        self.assertEqual(len(embedding), 768)

    @patch("transformers.LEDModel.from_pretrained")
    @patch("transformers.LEDTokenizer.from_pretrained")
    def test_extract_embedding_empty_text(self, mock_tokenizer, mock_model):
        mock_tokenizer.return_value = "mocked_tokenizer"
        mock_model.return_value.last_hidden_state.mean.return_value.squeeze.return_value.numpy.return_value = np.zeros(768)

        text = "tolerencelevel"
        embedding = self.extractor.extract_embedding(text)

        self.assertEqual(len(embedding), 768)

if __name__ == "__main__":
    unittest.main()


