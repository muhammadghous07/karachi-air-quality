from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained model with the corrected path for running from the 'api' folder
model = joblib.load('../models/basic_xgboost_model.pkl')

# Load feature columns used during training
feature_cols = [
    'temperature', 'humidity', 'wind_speed', 'precipitation', 
    'holiday', 'year', 'month', 'day', 'day_of_week', 'day_of_year',
    'week_of_year', 'quarter', 'month_sin', 'month_cos', 
    'day_of_week_sin', 'day_of_week_cos', 'is_weekend'
]

@app.route('/')
def home():
    return "Air Quality Prediction API for Karachi"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        
        # Convert date to datetime and extract features
        input_data['date'] = pd.to_datetime(input_data['date'])
        input_data['year'] = input_data['date'].dt.year
        input_data['month'] = input_data['date'].dt.month
        input_data['day'] = input_data['date'].dt.day
        input_data['day_of_week'] = input_data['date'].dt.dayofweek
        input_data['day_of_year'] = input_data['date'].dt.dayofyear
        
        # isocalendar().week returns a tuple, so we need to get the week number
        input_data['week_of_year'] = input_data['date'].dt.isocalendar().week.astype(int)
        
        input_data['quarter'] = input_data['date'].dt.quarter
        
        # Cyclical encoding
        input_data['month_sin'] = np.sin(2 * np.pi * input_data['month'] / 12)
        input_data['month_cos'] = np.cos(2 * np.pi * input_data['month'] / 12)
        input_data['day_of_week_sin'] = np.sin(2 * np.pi * input_data['day_of_week'] / 7)
        input_data['day_of_week_cos'] = np.cos(2 * np.pi * input_data['day_of_week'] / 7)
        
        # Weekend flag
        input_data['is_weekend'] = (input_data['day_of_week'] >= 5).astype(int)
        
        # Select only the features used during training
        # This also ensures the correct order
        input_data = input_data[feature_cols]
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Return prediction
        return jsonify({
            'prediction': float(prediction[0]),
            'message': 'PM2.5 prediction for Karachi'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'An error occurred during prediction. Please check your input data.',
            'expected_features': feature_cols
        })

@app.route('/health_impact', methods=['POST'])
def health_impact():
    try:
        data = request.get_json()
        pm25_level = data.get('pm25', 0)
        
        # Simple health impact estimation
        respiratory_risk = max(0, 0.5 * pm25_level + np.random.normal(10, 2))
        cardiovascular_risk = max(0, 0.3 * pm25_level + np.random.normal(5, 1))
        
        # Categorize air quality
        if pm25_level <= 50:
            aqi_category = "Good"
            health_advice = "Air quality is satisfactory, and air pollution poses little or no risk."
        elif pm25_level <= 100:
            aqi_category = "Moderate"
            health_advice = "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
        elif pm25_level <= 150:
            aqi_category = "Unhealthy for Sensitive Groups"
            health_advice = "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
        elif pm25_level <= 200:
            aqi_category = "Unhealthy"
            health_advice = "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
        else:
            aqi_category = "Very Unhealthy"
            health_advice = "Health alert: The risk of health effects is increased for everyone."
        
        return jsonify({
            'pm25_level': pm25_level,
            'aqi_category': aqi_category,
            'respiratory_risk': respiratory_risk,
            'cardiovascular_risk': cardiovascular_risk,
            'health_advice': health_advice
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 