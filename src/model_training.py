# File: src/model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

class ModelTrainer:
    def __init__(self):
        self.data = None
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def load_data(self):
        """Load engineered features"""
        self.data = pd.read_csv('../data/processed/engineered_features.csv', index_col='date', parse_dates=True)
        return self
    
    def prepare_data(self, target='pm25', test_size=0.2):
        """Prepare data for training with a fixed set of features"""
        print(f"Preparing data for target: {target}")
        
        # --- FIX: Define a fixed list of features that the frontend can provide ---
        feature_cols = [
            'temperature', 'humidity', 'wind_speed', 'precipitation', 
            'holiday', 'year', 'month', 'day', 'day_of_week', 'day_of_year',
            'week_of_year', 'quarter', 'month_sin', 'month_cos', 
            'day_of_week_sin', 'day_of_week_cos', 'is_weekend'
        ]
        
        # Ensure only the required features exist in the dataframe before proceeding
        # Create these features if they don't already exist in the dataframe
        self.data['year'] = self.data.index.year
        self.data['month'] = self.data.index.month
        self.data['day'] = self.data.index.day
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['day_of_year'] = self.data.index.dayofyear
        self.data['week_of_year'] = self.data.index.isocalendar().week.astype(int)
        self.data['quarter'] = self.data.index.quarter
        self.data['month_sin'] = np.sin(2 * np.pi * self.data['month'] / 12)
        self.data['month_cos'] = np.cos(2 * np.pi * self.data['month'] / 12)
        self.data['day_of_week_sin'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['day_of_week_cos'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['is_weekend'] = (self.data['day_of_week'] >= 5).astype(int)
        
        X = self.data[feature_cols]
        y = self.data[target]
        
        # Chronological split (time series)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_models(self, target='pm25'):
        """Train multiple models"""
        print(f"Training models for target: {target}")
        
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_data(target)
        
        # Define models
        models = {
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            # Store model
            self.models[name] = model
            
            # Store feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(feature_cols, model.feature_importances_))
            
            print(f"{name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        return self
    
    def save_models(self):
        """Save trained models to disk"""
        for name, model in self.models.items():
            # Naye model ko sahi jagah par save karen
            filename = f'../models/basic_xgboost_model.pkl'
            joblib.dump(model, filename)
            print(f"Saved {name} to {filename}")
        
        serializable_feature_importance = {}
        for model_name, importance_dict in self.feature_importance.items():
            serializable_feature_importance[model_name] = {
                feature: float(value) for feature, value in importance_dict.items()
            }
        
        with open('../models/model_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        
        with open('../models/feature_importance.json', 'w') as f:
            json.dump(serializable_feature_importance, f, indent=4)
        
        return self
    
    def plot_results(self, target='pm25'):
        """Plot model performance and predictions"""
        X_train, X_test, y_train, y_test, _ = self.prepare_data(target)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Actual vs Predicted for each model
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            axes[0, 0].scatter(y_test, y_pred, alpha=0.5, label=name)
        
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].legend()
        
        # Plot 2: Time series of actual vs predicted (XGBoost only)
        if 'XGBoost' in self.models:
            y_pred_xgb = self.models['XGBoost'].predict(X_test)
            axes[0, 1].plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
            axes[0, 1].plot(y_test.index, y_pred_xgb, label='Predicted (XGBoost)', alpha=0.7)
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel(target)
            axes[0, 1].set_title('Time Series: Actual vs Predicted')
            axes[0, 1].legend()
        
        # Plot 3: Model performance comparison
        model_names = list(self.results.keys())
        rmse_values = [self.results[name]['rmse'] for name in model_names]
        
        axes[1, 0].bar(model_names, rmse_values)
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('Model Comparison (RMSE)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Feature importance (XGBoost)
        if 'XGBoost' in self.feature_importance:
            feat_imp = pd.DataFrame({
                'feature': list(self.feature_importance['XGBoost'].keys()),
                'importance': list(self.feature_importance['XGBoost'].values())
            })
            feat_imp = feat_imp.sort_values('importance', ascending=False).head(10)
            
            axes[1, 1].barh(feat_imp['feature'], feat_imp['importance'])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Features (XGBoost)')
        
        plt.tight_layout()
        plt.savefig(f'../models/model_performance_{target}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self

# Run model training if script is executed directly
if __name__ == "__main__":
    # Create models directory
    os.makedirs('../models', exist_ok=True)
    
    # Train models for PM2.5 prediction
    trainer = ModelTrainer()
    trainer.load_data().train_models(target='pm25').save_models().plot_results(target='pm25')