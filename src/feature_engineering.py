import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.data = None
        
    def load_data(self):
        """Load processed data"""
        self.data = pd.read_csv('data/processed/merged_data.csv', index_col='date', parse_dates=True)
        print("Loaded processed data")
        return self
    
    def create_time_features(self):
        """Create time-based features"""
        print("Creating time-based features...")
        
        # Date components
        self.data['year'] = self.data.index.year
        self.data['month'] = self.data.index.month
        self.data['day'] = self.data.index.day
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['day_of_year'] = self.data.index.dayofyear
        self.data['week_of_year'] = self.data.index.isocalendar().week
        self.data['quarter'] = self.data.index.quarter
        
        # Cyclical encoding for periodic features
        self.data['month_sin'] = np.sin(2 * np.pi * self.data['month'] / 12)
        self.data['month_cos'] = np.cos(2 * np.pi * self.data['month'] / 12)
        self.data['day_of_week_sin'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['day_of_week_cos'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
        
        # Weekend flag
        self.data['is_weekend'] = (self.data['day_of_week'] >= 5).astype(int)
        
        return self
    
    def create_lag_features(self, lags=[1, 2, 3, 7, 14, 30]):
        """Create lagged features for pollutants"""
        print("Creating lag features...")
        
        pollutant_cols = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'aqi']
        
        for col in pollutant_cols:
            if col in self.data.columns:
                for lag in lags:
                    self.data[f'{col}_lag_{lag}'] = self.data[col].shift(lag)
        
        return self
    
    def create_rolling_features(self, windows=[7, 14, 30]):
        """Create rolling average features"""
        print("Creating rolling features...")
        
        pollutant_cols = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'aqi']
        
        for col in pollutant_cols:
            if col in self.data.columns:
                for window in windows:
                    self.data[f'{col}_rolling_mean_{window}'] = self.data[col].rolling(window=window).mean()
                    self.data[f'{col}_rolling_std_{window}'] = self.data[col].rolling(window=window).std()
        
        return self
    
    def handle_missing_values(self):
        """Handle missing values created by lag and rolling features"""
        print("Handling missing values...")
        
        # Drop rows with NaN values (created by lag/rolling features)
        self.data = self.data.dropna()
        
        return self
    
    def save_features(self):
        """Save engineered features to file"""
        self.data.to_csv('data/processed/engineered_features.csv')
        print("Engineered features saved to data/processed/engineered_features.csv")
        return self

# Run feature engineering if script is executed directly
if __name__ == "__main__":
    engineer = FeatureEngineer()
    (engineer.load_data()
             .create_time_features()
             .create_lag_features()
             .create_rolling_features()
             .handle_missing_values()
             .save_features())