import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self):
        self.aqi_data = None
        self.weather_data = None
        self.health_data = None
        self.holidays_data = None
        self.merged_data = None
        
    def load_data(self):
        """Load all raw data files"""
        self.aqi_data = pd.read_csv('data/raw/aqi_data.csv', parse_dates=['date'])
        self.weather_data = pd.read_csv('data/raw/weather_data.csv', parse_dates=['date'])
        self.health_data = pd.read_csv('data/raw/health_data.csv', parse_dates=['date'])
        self.holidays_data = pd.read_csv('data/external/holidays.csv', parse_dates=['date'])
        
        return self
    
    def clean_data(self):
        """Clean and preprocess the data"""
        print("Cleaning data...")
        
        # Handle missing values
        self.aqi_data = self.aqi_data.fillna(method='ffill')
        self.weather_data = self.weather_data.fillna(method='ffill')
        self.health_data = self.health_data.fillna(method='ffill')
        
        # Remove duplicates
        self.aqi_data = self.aqi_data.drop_duplicates(subset=['date'])
        self.weather_data = self.weather_data.drop_duplicates(subset=['date'])
        self.health_data = self.health_data.drop_duplicates(subset=['date'])
        
        # Ensure date is the index
        self.aqi_data.set_index('date', inplace=True)
        self.weather_data.set_index('date', inplace=True)
        self.health_data.set_index('date', inplace=True)
        
        return self
    
    def merge_data(self):
        """Merge all datasets on date"""
        print("Merging data...")
        
        # Merge AQI and weather data
        self.merged_data = self.aqi_data.merge(
            self.weather_data, 
            left_index=True, 
            right_index=True, 
            how='outer'
        )
        
        # Merge with health data
        self.merged_data = self.merged_data.merge(
            self.health_data,
            left_index=True,
            right_index=True,
            how='outer'
        )
        
        # Add holiday information
        self.holidays_data.set_index('date', inplace=True)
        self.merged_data = self.merged_data.merge(
            self.holidays_data,
            left_index=True,
            right_index=True,
            how='left'
        )
        
        # Fill NaN for holidays (0 = not holiday, 1 = holiday)
        self.merged_data['holiday'] = self.merged_data['holiday'].notna().astype(int)
        
        # Forward fill any remaining missing values
        self.merged_data = self.merged_data.fillna(method='ffill')
        
        return self
    
    def save_processed_data(self):
        """Save processed data to file"""
        self.merged_data.to_csv('data/processed/merged_data.csv')
        print("Processed data saved to data/processed/merged_data.csv")
        
        return self

# Run data processing if script is executed directly
if __name__ == "__main__":
    processor = DataProcessor()
    processor.load_data().clean_data().merge_data().save_processed_data()