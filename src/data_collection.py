import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

load_dotenv()

class DataCollector:
    def __init__(self):
        self.aqi_data = None
        self.weather_data = None
        self.health_data = None
        self.holidays_data = None
        
    def fetch_aqi_data(self, city="Karachi", days=365):
        """
        Fetch AQI data from OpenAQ API (simulated with synthetic data)
        In a real scenario, you would use actual API calls
        """
        print("Fetching AQI data...")
        
        # Simulate API call delay
        time.sleep(1)
        
        # Generate synthetic data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Simulate seasonal patterns
        base_pm25 = np.random.normal(60, 15, days)
        seasonal_effect = 15 * np.sin(2 * np.pi * np.arange(days) / 365)
        trend = 0.05 * np.arange(days)
        
        pm25 = base_pm25 + seasonal_effect + trend
        pm10 = pm25 * 1.5 + np.random.normal(0, 5, days)
        no2 = np.random.normal(30, 10, days)
        so2 = np.random.normal(15, 5, days)
        o3 = np.random.normal(40, 15, days)
        
        # Ensure values are positive
        pm25 = np.abs(pm25)
        pm10 = np.abs(pm10)
        no2 = np.abs(no2)
        so2 = np.abs(so2)
        o3 = np.abs(o3)
        
        self.aqi_data = pd.DataFrame({
            'date': dates,
            'pm25': pm25,
            'pm10': pm10,
            'no2': no2,
            'so2': so2,
            'o3': o3
        })
        
        # Calculate AQI (simplified)
        self.aqi_data['aqi'] = self.aqi_data['pm25'] * 100 / 50  # Simplified calculation
        
        # Save to CSV
        self.aqi_data.to_csv('data/raw/aqi_data.csv', index=False)
        print("AQI data saved to data/raw/aqi_data.csv")
        
        return self.aqi_data
    
    def fetch_weather_data(self, city="Karachi", days=365):
        """
        Fetch weather data from Open-Meteo API (simulated with synthetic data)
        """
        print("Fetching weather data...")
        
        # Simulate API call delay
        time.sleep(1)
        
        # Generate synthetic data
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Simulate seasonal weather patterns for Karachi
        base_temp = np.random.normal(28, 5, days)
        seasonal_effect = 8 * np.sin(2 * np.pi * np.arange(days) / 365)
        temperature = base_temp + seasonal_effect
        
        humidity = np.random.normal(65, 15, days)
        wind_speed = np.random.normal(12, 5, days)
        precipitation = np.random.exponential(0.5, days)
        
        # Ensure values are within reasonable ranges
        temperature = np.clip(temperature, 10, 45)
        humidity = np.clip(humidity, 20, 100)
        wind_speed = np.clip(wind_speed, 0, 30)
        precipitation = np.clip(precipitation, 0, 50)
        
        self.weather_data = pd.DataFrame({
            'date': dates,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'precipitation': precipitation
        })
        
        # Save to CSV
        self.weather_data.to_csv('data/raw/weather_data.csv', index=False)
        print("Weather data saved to data/raw/weather_data.csv")
        
        return self.weather_data
    
    def generate_health_data(self, days=365):
        """
        Generate synthetic health data correlated with AQI
        """
        print("Generating health data...")
        
        if self.aqi_data is None:
            self.fetch_aqi_data(days=days)
        
        dates = self.aqi_data['date']
        pm25 = self.aqi_data['pm25']
        
        # Generate health metrics correlated with PM2.5
        respiratory_cases = np.random.poisson(50 + pm25 * 0.5)
        cardiovascular_cases = np.random.poisson(30 + pm25 * 0.3)
        mortality_rate = np.random.normal(0.5 + pm25 * 0.01, 0.1)
        
        # Ensure reasonable values
        respiratory_cases = np.clip(respiratory_cases, 0, 200)
        cardiovascular_cases = np.clip(cardiovascular_cases, 0, 150)
        mortality_rate = np.clip(mortality_rate, 0.3, 2.0)
        
        self.health_data = pd.DataFrame({
            'date': dates,
            'respiratory_cases': respiratory_cases,
            'cardiovascular_cases': cardiovascular_cases,
            'mortality_rate': mortality_rate
        })
        
        # Save to CSV
        self.health_data.to_csv('data/raw/health_data.csv', index=False)
        print("Health data saved to data/raw/health_data.csv")
        
        return self.health_data
    
    def fetch_holidays(self, year=2023):
        """
        Compile holiday list for Pakistan
        """
        print("Compiling holiday data...")
        
        # Major holidays in Pakistan (fixed dates)
        holidays = [
            f"{year}-01-01",  # New Year's Day
            f"{year}-03-23",  # Pakistan Day
            f"{year}-05-01",  # Labour Day
            f"{year}-08-14",  # Independence Day
            f"{year}-09-06",  # Defense Day
            f"{year}-12-25",  # Quaid-e-Azam Day
            f"{year}-12-31",  # New Year's Eve
        ]
        
        # Islamic holidays (approximate dates)
        eid_al_fitr = f"{year}-04-22"  # Approximate
        eid_al_adha = f"{year}-06-29"  # Approximate
        muharram = f"{year}-07-29"     # Approximate
        
        holidays.extend([eid_al_fitr, eid_al_adha, muharram])
        
        self.holidays_data = pd.DataFrame({
            'date': pd.to_datetime(holidays),
            'holiday': 'public_holiday'
        })
        
        # Save to CSV
        self.holidays_data.to_csv('data/external/holidays.csv', index=False)
        print("Holiday data saved to data/external/holidays.csv")
        
        return self.holidays_data
    
    def collect_all_data(self):
        """Collect all required data"""
        self.fetch_aqi_data()
        self.fetch_weather_data()
        self.generate_health_data()
        self.fetch_holidays()
        print("All data collection complete!")

# Run data collection if script is executed directly
if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_all_data()