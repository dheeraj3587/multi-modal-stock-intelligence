import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator
import joblib
import os
import json
from typing import Dict, Any, Optional, List, Union, Tuple

class GrowthScorer:
    """
    Growth scoring ensemble combining fundamental and technical indicators.
    Predicts 60-90 day forward returns to rank stocks.
    
    Aligned with docs/metrics_and_evaluation.md Section 3 targets:
    - Spearman Rank Correlation >= 0.30
    - Top-10 Precision >= 70%
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the growth scorer.
        
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split an internal node
            learning_rate: Learning rate (for Gradient Boosting)
            random_state: Random seed
        """
        self.model_type = model_type
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'random_state': random_state
        }
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(**self.params)
        elif model_type == 'gradient_boosting':
            self.params['learning_rate'] = learning_rate
            self.model = GradientBoostingRegressor(**self.params)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
            
        self.feature_names = []
        self.scalers = None  # Dict containing all scalers (fundamental, technical, sector_medians)
        
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series], 
        feature_names: Optional[List[str]] = None,
        scalers: Optional[Dict] = None
    ):
        """
        Fit the model on training data.
        
        Args:
            X: Feature matrix
            y: Target vector (forward returns)
            feature_names: List of feature names (optional if X is DataFrame)
            scalers: Dict containing scalers (fundamental, technical, sector_medians) used to preprocess X
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        elif feature_names:
            self.feature_names = feature_names
            
        if isinstance(y, pd.Series):
            y = y.values
            
        self.validate_features(X)
        self.model.fit(X, y)
        
        if scalers:
            self.scalers = scalers
            
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict growth scores (forward returns).
        
        Args:
            X: Feature matrix
            
        Returns:
            scores: Predicted forward returns
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)
    
    def rank_stocks(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Rank stocks by predicted growth potential.
        
        Args:
            X: Feature matrix
            tickers: List of tickers corresponding to X rows
            
        Returns:
            DataFrame with columns [ticker, score, rank]
        """
        scores = self.predict(X)
        
        df = pd.DataFrame({
            'ticker': tickers,
            'score': scores
        })
        
        # Rank descending (1 = highest score)
        df['rank'] = df['score'].rank(ascending=False, method='min').astype(int)
        df = df.sort_values('rank')
        
        return df
    
    def get_feature_importances(self) -> pd.DataFrame:
        """
        Get feature importances.
        
        Returns:
            DataFrame with columns [feature, importance]
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model has not been fitted yet")
            
        importances = self.model.feature_importances_
        
        # Handle case where feature_names might not be set or match length
        names = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(len(importances))]
        
        if len(names) != len(importances):
             names = [f"feature_{i}" for i in range(len(importances))]
             
        df = pd.DataFrame({
            'feature': names,
            'importance': importances
        })
        
        return df.sort_values('importance', ascending=False)
    
    def validate_features(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Validate input features.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names to check against
        """
        if np.isnan(X).any():
            raise ValueError("Input contains NaN values")
            
        if np.isinf(X).any():
            raise ValueError("Input contains infinite values")
            
        if feature_names and len(feature_names) != X.shape[1]:
            raise ValueError(f"Feature count mismatch: expected {len(feature_names)}, got {X.shape[1]}")

    def save_checkpoint(self, path: str, metadata: Dict[str, Any]):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save .pkl file
            metadata: Metadata dictionary
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'feature_names': self.feature_names,
            'scalers': self.scalers,  # Save full scalers dict instead of single scaler
            'metadata': metadata
        }
        
        joblib.dump(checkpoint, path)
        
        # Save metadata as JSON
        json_path = path.replace('.pkl', '.json')
        with open(json_path, 'w') as f:
            # Filter non-serializable items
            serializable_metadata = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool, list, dict))}
            json.dump(serializable_metadata, f, indent=4)
            
    @classmethod
    def load_checkpoint(cls, path: str) -> Tuple['GrowthScorer', Dict[str, Any]]:
        """
        Load model from checkpoint.
        
        Args:
            path: Path to .pkl file
            
        Returns:
            model: Loaded GrowthScorer instance
            metadata: Metadata dictionary
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
            
        checkpoint = joblib.load(path)
        
        # Initialize instance
        instance = cls(
            model_type=checkpoint.get('model_type', 'random_forest'),
            **checkpoint.get('params', {})
        )
        
        # Restore state
        instance.model = checkpoint['model']
        instance.feature_names = checkpoint.get('feature_names', [])
        instance.scalers = checkpoint.get('scalers')
        
        return instance, checkpoint.get('metadata', {})
