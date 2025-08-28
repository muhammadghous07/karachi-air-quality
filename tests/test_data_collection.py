import pytest
import pandas as pd
from src.data_collection import DataCollector

def test_data_collection():
    """Test that data collection works correctly"""
    collector = DataCollector()
    
    # Test AQI data collection
    aqi_data = collector.fetch_aqi_data(days=10)
    assert aqi_data is not None
    assert len(aqi_data) == 10
    assert 'pm25' in aqi_data.columns
    
    # Test weather data collection
    weather_data = collector.fetch_weather_data(days=10)
    assert weather_data is not None
    assert len(weather_data) == 10
    assert 'temperature' in weather_data.columns
    
    print("Data collection tests passed!")

if __name__ == "__main__":
    test_data_collection()