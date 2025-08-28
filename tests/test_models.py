import pytest
import pandas as pd
from src.model_training import ModelTrainer

def test_model_training():
    """Test that model training works correctly"""
    trainer = ModelTrainer()
    trainer.load_data()
    
    # Test with small subset of data
    X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(target='pm25')
    
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None
    assert len(feature_cols) > 0
    
    print("Model training tests passed!")

if __name__ == "__main__":
    test_model_training()