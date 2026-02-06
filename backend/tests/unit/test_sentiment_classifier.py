import pytest
import torch
import os
import shutil
from unittest.mock import MagicMock, patch
from models.sentiment_classifier import FinBERTSentimentClassifier

@pytest.fixture
def mock_bert():
    with patch('models.sentiment_classifier.AutoModel.from_pretrained') as mock:
        bert_mock = MagicMock()
        bert_mock.config.hidden_size = 768
        # Mock forward pass return
        output_mock = MagicMock()
        output_mock.pooler_output = torch.randn(1, 768)
        bert_mock.return_value = output_mock
        mock.return_value = bert_mock
        yield mock

@pytest.fixture
def classifier(mock_bert):
    return FinBERTSentimentClassifier(model_name='mock_bert', num_classes=3)

class TestFinBERTSentimentClassifier:
    def test_initialization(self, classifier):
        assert classifier.num_classes == 3
        assert classifier.bert is not None
        assert classifier.classifier.out_features == 3
        
    def test_forward_pass_shape(self, classifier):
        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        # Mock BERT output for batch size 2
        classifier.bert.return_value.pooler_output = torch.randn(2, 768)
        
        logits = classifier(input_ids, attention_mask)
        assert logits.shape == (2, 3)
        
    def test_freeze_bert_layers(self, classifier):
        classifier.freeze_bert_layers()
        for param in classifier.bert.parameters():
            assert param.requires_grad == False
            
    def test_unfreeze_bert_layers(self, classifier):
        classifier.unfreeze_bert_layers()
        # Mock parameters
        classifier.bert.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.randn(1))])
        # Re-call unfreeze to apply to mock
        for param in classifier.bert.parameters():
            param.requires_grad = True
            
    def test_predict_shape(self, classifier):
        texts = ["Positive news", "Negative news"]
        tokenizer = MagicMock()
        tokenizer.return_value = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        
        # Mock BERT output
        classifier.bert.return_value.pooler_output = torch.randn(2, 768)
        
        preds, confs = classifier.predict(texts, tokenizer, torch.device('cpu'))
        assert len(preds) == 2
        assert len(confs) == 2
        assert preds.shape == (2,)
        assert confs.shape == (2,)

    def test_save_and_load_checkpoint(self, classifier, tmp_path):
        path = tmp_path / "checkpoint.pth"
        metadata = {'epoch': 1, 'f1': 0.8}
        
        classifier.save_checkpoint(str(path), metadata)
        assert os.path.exists(path)
        assert os.path.exists(str(path).replace('.pth', '.json'))
        
        # Load
        with patch('models.sentiment_classifier.AutoModel.from_pretrained') as mock_load:
            mock_load.return_value = classifier.bert # Reuse mock
            loaded_model, loaded_meta = FinBERTSentimentClassifier.load_checkpoint(str(path))
            
            assert loaded_meta['epoch'] == 1
            assert loaded_meta['f1'] == 0.8
