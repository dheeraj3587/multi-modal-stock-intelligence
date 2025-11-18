from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from scripts.text_embeddings import (
    align_to_trading_day,
    prepare_news_samples,
    prepare_social_samples,
    extract_embeddings_batch,
    generate_embeddings,
    select_device,
    load_models,
    process_embeddings,
)


def test_align_to_trading_day_weekend():
    monday = align_to_trading_day("2024-11-16T12:00:00Z")
    assert monday == "2024-11-18"


def test_prepare_news_samples_concatenates_text():
    records = [
        {"title": "Earnings beat", "description": "Strong quarter", "published_at": "2024-10-10T10:00:00Z", "url": "id1"},
        {"title": None, "description": None, "published_at": "2024-10-11T10:00:00Z", "url": "id2"},
    ]
    samples = prepare_news_samples(records, ticker="RELIANCE")
    assert len(samples) == 1
    assert "Earnings beat" in samples[0]["text"]
    assert samples[0]["aligned_day"] == "2024-10-10"


def test_prepare_social_samples_requires_timestamp():
    records = [{"messages": [{"body": "Test body"}]}]
    with pytest.raises(ValueError):
        prepare_social_samples(records, ticker="RELIANCE")


class DummyTokenizer:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        batch = len(texts)
        seq_len = min(max_length, 8)
        tensor = torch.ones((batch, seq_len), dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": tensor}


class DummyEmbedModel(torch.nn.Module):
    def forward(self, **kwargs):
        batch = kwargs["input_ids"].shape[0]
        seq_len = kwargs["input_ids"].shape[1]
        hidden = torch.arange(batch * seq_len * 4, dtype=torch.float32).view(batch, seq_len, 4)
        return SimpleNamespace(last_hidden_state=hidden)


class DummyClassifier(torch.nn.Module):
    def forward(self, **kwargs):
        batch = kwargs["input_ids"].shape[0]
        logits = torch.linspace(0.1, 0.9, steps=batch * 3, dtype=torch.float32).view(batch, 3)
        return SimpleNamespace(logits=logits)


def test_extract_embeddings_batch_outputs_shapes():
    tokenizer = DummyTokenizer()
    embed_model = DummyEmbedModel()
    classifier = DummyClassifier()
    batch = [{"text": "Stock up"}, {"text": "Stock down"}]
    embeddings, labels, probs = extract_embeddings_batch(
        batch=batch,
        tokenizer=tokenizer,
        embed_model=embed_model,
        classifier=classifier,
        label_map={0: "positive", 1: "negative", 2: "neutral"},
        device=torch.device("cpu"),
        max_length=5,
    )
    assert embeddings.shape == (2, 4)
    assert len(labels) == 2
    assert len(probs) == 2 and len(probs[0]) == 3


def test_generate_embeddings_accumulates_batches():
    tokenizer = DummyTokenizer()
    embed_model = DummyEmbedModel()
    classifier = None
    samples = [{"text": f"Sample {i}"} for i in range(5)]
    embeddings, labels, probs = generate_embeddings(
        samples,
        tokenizer=tokenizer,
        embed_model=embed_model,
        classifier=classifier,
        label_map=None,
        device=torch.device("cpu"),
        batch_size=2,
        max_length=6,
    )
    assert embeddings.shape[0] == 5
    assert embeddings.shape[1] == 4
    assert all(label is None for label in labels)
    assert all(prob is None for prob in probs)


class TestSelectDevice:
    """Test suite for select_device() function with CUDA validation."""

    def test_select_device_auto_cpu(self):
        """Test auto-selection falls back to CPU when CUDA unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            device = select_device(preferred=None)
            assert device.type == "cpu"

    def test_select_device_auto_cuda(self):
        """Test auto-selection uses CUDA when available."""
        with patch("torch.cuda.is_available", return_value=True):
            device = select_device(preferred=None)
            assert device.type == "cuda"

    def test_select_device_explicit_cpu(self):
        """Test explicit CPU selection."""
        device = select_device(preferred="cpu")
        assert device.type == "cpu"

    def test_select_device_cuda_requested_but_unavailable(self):
        """Test CUDA requested but unavailable falls back to CPU with warning."""
        with patch("torch.cuda.is_available", return_value=False):
            device = select_device(preferred="cuda")
            assert device.type == "cpu", "Should fall back to CPU when CUDA unavailable"

    def test_select_device_cuda_requested_and_available(self):
        """Test CUDA requested and available returns CUDA device."""
        with patch("torch.cuda.is_available", return_value=True):
            device = select_device(preferred="cuda")
            assert device.type == "cuda"

    def test_select_device_cuda_index_unavailable(self):
        """Test CUDA with device index (cuda:0) falls back when unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            device = select_device(preferred="cuda:0")
            assert device.type == "cpu", "Should fall back to CPU for cuda:0 when unavailable"

    def test_select_device_cuda_index_available(self):
        """Test CUDA with device index (cuda:0) when available."""
        with patch("torch.cuda.is_available", return_value=True):
            device = select_device(preferred="cuda:0")
            assert device.type == "cuda"


class TestModelLoadingErrors:
    """Test suite for model loading error handling."""

    def test_load_models_oserror_handling(self):
        """Test that OSError during model loading is handled gracefully."""
        with patch("scripts.text_embeddings.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.side_effect = OSError("Connection timeout")
            
            with pytest.raises(OSError, match="Connection timeout"):
                load_models(model_name="ProsusAI/finbert", device=torch.device("cpu"), classify=False)

    def test_load_models_valueerror_handling(self):
        """Test that ValueError during model loading is handled gracefully."""
        with patch("scripts.text_embeddings.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.side_effect = ValueError("Invalid model configuration")
            
            with pytest.raises(ValueError, match="Invalid model configuration"):
                load_models(model_name="invalid-model", device=torch.device("cpu"), classify=False)

    def test_process_embeddings_model_load_failure_oserror(self, tmp_path):
        """Test process_embeddings catches and re-raises OSError with helpful message."""
        # Create minimal test data
        input_dir = tmp_path / "input" / "20241118_120000"
        input_dir.mkdir(parents=True)
        test_file = input_dir / "TEST_news.json"
        test_file.write_text('[{"title": "Test", "description": "Test", "published_at": "2024-11-18T10:00:00Z", "url": "test"}]')
        
        with patch("scripts.text_embeddings.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.side_effect = OSError("Failed to download model files")
            
            with pytest.raises(RuntimeError, match="Model loading failed"):
                process_embeddings(
                    ticker="TEST",
                    source="news",
                    input_dir=tmp_path / "input",
                    output_dir=tmp_path / "output",
                    model_name="ProsusAI/finbert",
                    batch_size=16,
                    max_length=512,
                    device_str="cpu",
                    classify_sentiment=False,
                )

    def test_process_embeddings_model_load_failure_valueerror(self, tmp_path):
        """Test process_embeddings catches and re-raises ValueError with helpful message."""
        # Create minimal test data
        input_dir = tmp_path / "input" / "20241118_120000"
        input_dir.mkdir(parents=True)
        test_file = input_dir / "TEST_news.json"
        test_file.write_text('[{"title": "Test", "description": "Test", "published_at": "2024-11-18T10:00:00Z", "url": "test"}]')
        
        with patch("scripts.text_embeddings.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.side_effect = ValueError("Invalid model name format")
            
            with pytest.raises(RuntimeError, match="Model loading failed"):
                process_embeddings(
                    ticker="TEST",
                    source="news",
                    input_dir=tmp_path / "input",
                    output_dir=tmp_path / "output",
                    model_name="invalid/model",
                    batch_size=16,
                    max_length=512,
                    device_str="cpu",
                    classify_sentiment=False,
                )

    def test_load_models_success(self):
        """Test successful model loading returns expected components."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        with patch("scripts.text_embeddings.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("scripts.text_embeddings.AutoModel.from_pretrained", return_value=mock_model):
            
            tokenizer, embed_model, classifier, label_map = load_models(
                model_name="ProsusAI/finbert",
                device=torch.device("cpu"),
                classify=False
            )
            
            assert tokenizer is mock_tokenizer
            assert embed_model is mock_model
            assert classifier is None
            assert label_map is None
            mock_model.to.assert_called_once()
            mock_model.eval.assert_called_once()

