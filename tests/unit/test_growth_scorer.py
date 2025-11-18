import pytest
import numpy as np
import pandas as pd
import os
from models.growth_scorer import GrowthScorer

@pytest.fixture
def sample_data():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    feature_names = [f'feat_{i}' for i in range(5)]
    return X, y, feature_names

@pytest.fixture
def scorer():
    return GrowthScorer(n_estimators=10, max_depth=3)

class TestGrowthScorer:
    def test_fit_and_predict(self, scorer, sample_data):
        X, y, names = sample_data
        scorer.fit(X, y, feature_names=names)
        preds = scorer.predict(X)
        
        assert len(preds) == 100
        assert scorer.feature_names == names
        
    def test_rank_stocks(self, scorer, sample_data):
        X, y, _ = sample_data
        scorer.fit(X, y)
        tickers = [f'TICK_{i}' for i in range(100)]
        
        ranked_df = scorer.rank_stocks(X, tickers)
        
        assert len(ranked_df) == 100
        assert 'rank' in ranked_df.columns
        assert ranked_df['rank'].min() == 1
        assert ranked_df['rank'].max() == 100
        # Check sorting
        assert ranked_df.iloc[0]['rank'] == 1
        
    def test_feature_importances(self, scorer, sample_data):
        X, y, names = sample_data
        scorer.fit(X, y, feature_names=names)
        
        imp_df = scorer.get_feature_importances()
        assert len(imp_df) == 5
        assert 'importance' in imp_df.columns
        assert imp_df['feature'].tolist() != []
        
    def test_save_and_load_checkpoint(self, scorer, sample_data, tmp_path):
        X, y, names = sample_data
        scorer.fit(X, y, feature_names=names)
        
        path = tmp_path / "growth_model.pkl"
        metadata = {'test_metric': 0.5}
        
        scorer.save_checkpoint(str(path), metadata)
        assert os.path.exists(path)
        
        loaded_scorer, loaded_meta = GrowthScorer.load_checkpoint(str(path))
        assert loaded_meta['test_metric'] == 0.5
        assert loaded_scorer.feature_names == names
        
    def test_validate_features(self, scorer):
        X_nan = np.array([[1, np.nan], [2, 3]])
        with pytest.raises(ValueError, match="NaN"):
            scorer.validate_features(X_nan)
            
        X_inf = np.array([[1, np.inf], [2, 3]])
        with pytest.raises(ValueError, match="infinite"):
            scorer.validate_features(X_inf)
