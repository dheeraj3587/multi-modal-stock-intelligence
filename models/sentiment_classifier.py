import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os
import json
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

class FinBERTSentimentClassifier(nn.Module):
    """
    FinBERT-based sentiment classifier for 3-class financial sentiment analysis.
    Classes: 0 (Positive), 1 (Neutral), 2 (Negative).
    
    Aligned with docs/metrics_and_evaluation.md Section 2 targets:
    - Macro F1 >= 0.80
    - Per-class Precision/Recall >= 0.75
    """
    
    def __init__(
        self, 
        model_name: str = "ProsusAI/finbert", 
        num_classes: int = 3, 
        dropout: float = 0.3,
        freeze_bert: bool = True
    ):
        """
        Initialize the sentiment classifier.
        
        Args:
            model_name: Hugging Face model identifier (default: ProsusAI/finbert)
            num_classes: Number of sentiment classes (default: 3)
            dropout: Dropout probability for the classification head
            freeze_bert: Whether to freeze BERT layers initially
        """
        super(FinBERTSentimentClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained FinBERT
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # Freeze BERT layers if requested
        if freeze_bert:
            self.freeze_bert_layers()
            
    def freeze_bert_layers(self):
        """Freeze all BERT parameters."""
        for param in self.bert.parameters():
            param.requires_grad = False
            
    def unfreeze_bert_layers(self):
        """Unfreeze all BERT parameters for fine-tuning."""
        for param in self.bert.parameters():
            param.requires_grad = True
            
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: Raw output logits [batch_size, num_classes]
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token embedding (pooler_output)
        # Note: pooler_output includes a dense layer + tanh activation in standard BERT
        # We can also use last_hidden_state[:, 0, :]
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification layer
        logits = self.classifier(pooled_output)
        
        return logits
    
    def predict(
        self, 
        texts: Union[str, List[str]], 
        tokenizer: Any, 
        device: torch.device, 
        max_length: int = 512,
        batch_size: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on texts.
        
        Args:
            texts: Single string or list of strings
            tokenizer: Hugging Face tokenizer
            device: Torch device
            max_length: Max token length
            batch_size: Batch size for inference
            
        Returns:
            predicted_classes: Array of class indices [num_samples]
            confidence_scores: Array of confidence scores (max probability) [num_samples]
        """
        self.eval()
        if isinstance(texts, str):
            texts = [texts]
            
        all_preds = []
        all_confs = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            with torch.no_grad():
                logits = self(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                
                confs, preds = torch.max(probs, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_confs.append(confs.cpu().numpy())
                
        return np.concatenate(all_preds), np.concatenate(all_confs)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total_params, "trainable": trainable_params}
    
    def save_checkpoint(
        self, 
        path: str, 
        metadata: Dict[str, Any], 
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Save model checkpoint and metadata.
        
        Args:
            path: Path to save .pth file
            metadata: Dictionary of metadata to save
            optimizer: Optional optimizer to save state
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare state dict
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'metadata': metadata,
            'config': {
                'model_name': self.model_name,
                'num_classes': self.num_classes
            }
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        # Save .pth
        torch.save(checkpoint, path)
        
        # Save human-readable metadata
        json_path = path.replace('.pth', '.json')
        with open(json_path, 'w') as f:
            # Filter non-serializable items if necessary
            serializable_metadata = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool, list, dict))}
            json.dump(serializable_metadata, f, indent=4)
            
    @classmethod
    def load_checkpoint(cls, path: str, device: torch.device = torch.device('cpu')) -> Tuple['FinBERTSentimentClassifier', Dict[str, Any]]:
        """
        Load model from checkpoint.
        
        Args:
            path: Path to .pth file
            device: Device to load model to
            
        Returns:
            model: Loaded FinBERTSentimentClassifier
            metadata: Loaded metadata dict
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
            
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get('config', {})
        
        # Initialize model
        model = cls(
            model_name=config.get('model_name', "ProsusAI/finbert"),
            num_classes=config.get('num_classes', 3),
            freeze_bert=False # Unfreeze to load weights correctly
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model, checkpoint.get('metadata', {})
