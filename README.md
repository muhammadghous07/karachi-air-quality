# 🌤️ Karachi Air Quality & Health Prediction System

A comprehensive web-based application that predicts PM2.5 air quality levels and assesses their health impacts in Karachi, Pakistan. This project features a decoupled architecture with a Flask API backend and Streamlit frontend, powered by an XGBoost Regressor model trained on meteorological and temporal data.

![Python](https://img.shields.io/badge/Python-3.11%252B-blue)
![ML](https://img.shields.io/badge/ML-XGBoost-orange)
![API](https://img.shields.io/badge/API-Flask-green)
![UI](https://img.shields.io/badge/UI-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📋 Table of Contents
- [🌤️ Karachi Air Quality \& Health Prediction System](#️-karachi-air-quality--health-prediction-system)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Project Overview](#-project-overview)
  - [🏗️ Architecture](#️-architecture)
  - [✨ Features](#-features)
    - [🔬 Core Functionality](#-core-functionality)
    - [🎨 User Interface](#-user-interface)
    - [⚙️ Technical Features](#️-technical-features)
  - [💻 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [Step-by-Step Setup](#step-by-step-setup)
- [1. Clone the repository](#1-clone-the-repository)
- [2. Create virtual environment](#2-create-virtual-environment)
- [3. Activate virtual environment](#3-activate-virtual-environment)
- [On Windows:](#on-windows)
- [On macOS/Linux:](#on-macoslinux)
- [4. Install dependencies](#4-install-dependencies)
- [Core Data Science](#core-data-science)
- [API \& Web Framework](#api--web-framework)
- [Utilities \& Visualization](#utilities--visualization)

---

## 🎯 Project Overview
This project addresses Karachi's air quality challenges by providing:
- **Real-time PM2.5 predictions** based on weather and temporal data  
- **Health impact assessments** with actionable advice  
- **Interactive web interface** for easy access and visualization  
- **Machine learning backend** with XGBoost regression model  
- **RESTful API** for seamless integration with other applications  

---

## 🏗️ Architecture
The system follows a modern decoupled architecture:

```text
┌─────────────────┐    HTTP Requests    ┌─────────────────┐
│                 │ ◄────────────────── │                 │
│  Streamlit      │                     │   Flask API     │
│  Frontend       │ ──────────────────► │   Backend       │
│  (UI Layer)     │    JSON Responses   │   (Logic Layer) │
│                 │                     │                 │
└─────────────────┘                     └─────────────────┘
         │                                          │
         │                                          │
         ▼                                          ▼
┌─────────────────┐                     ┌─────────────────┐
│   User          │                     │   ML Model      │
│   Browser       │                     │   (XGBoost)     │
│                 │                     │                 │
└─────────────────┘                     └─────────────────┘


---

## ✨ Features

### 🔬 Core Functionality
- PM2.5 Prediction: Accurate air quality forecasting using XGBoost  
- Health Impact Assessment: Comprehensive health risk evaluation  
- Real-time Data Processing: Instant predictions based on user input  
- Multi-model Support: XGBoost, Random Forest, and Linear Regression  

### 🎨 User Interface
- Interactive Dashboard: Streamlit-based responsive design  
- User-friendly Inputs: Sliders, date pickers, and selection boxes  
- Visual Feedback: Color-coded results and health recommendations  
- Mobile Responsive: Works on desktop and mobile devices  

### ⚙️ Technical Features
- RESTful API: Clean JSON API with proper error handling  
- Model Persistence: Joblib-serialized machine learning models  
- Feature Engineering: Advanced temporal and weather features  
- Comprehensive Testing: Unit tests for all components  

---

## 💻 Installation

### Prerequisites
- Python 3.11 or higher  
- pip package manager  
- Git  

### Step-by-Step Setup
```bash
# 1. Clone the repository
git clone https://github.com/your-username/karachi-air-quality.git
cd karachi-air-quality

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# Core Data Science
pandas==2.1.3
numpy==1.26.0
scikit-learn==1.3.2
xgboost==2.0.3

# API & Web Framework
flask==2.3.3
streamlit==1.28.0
requests==2.31.0

# Utilities & Visualization
matplotlib==3.8.0
seaborn==0.13.0
joblib==1.3.2
python-dotenv==1.0.0

🚀 Quick Start

1. Train the Machine Learning Model

cd src
python model_training.py

This will create:

models/basic_xgboost_model.pkl (trained model)

models/model_results.json (performance metrics)

models/model_performance_pm25.png (visualization)

2. Start the Backend API Server

cd api
flask --app app run --host=0.0.0.0 --port=5000

3. Launch the Frontend Application

cd frontend
streamlit run app_streamlit.py

Open in browser: http://localhost:8501

📁 Project Structure

karachi-air-quality/
├── api/                  # Flask Backend API
│   ├── app.py
│   ├── requirements.txt
│   └── test_api.py
├── frontend/             # Streamlit Frontend
│   ├── app_streamlit.py
│   └── requirements.txt
├── src/                  # ML Training & Processing
│   ├── data_collection.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   └── model_training.py
├── models/               # Trained Models & Results
│   ├── basic_xgboost_model.pkl
│   ├── random_forest_model.pkl
│   ├── linear_regression_model.pkl
│   ├── model_results.json
│   └── model_performance_pm25.png
├── data/                 # Data Storage
│   ├── raw/
│   ├── processed/
│   └── external/
├── tests/                # Test Suite
│   ├── test_data_collection.py
│   ├── test_data_processing.py
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   └── test_api.py
├── requirements.txt
└── README.md

🌐 API Documentation

Base URL: http://localhost:5000

1. Predict PM2.5 Levels

POST /predict
Request:

{
  "date": "2023-12-15",
  "temperature": 25,
  "humidity": 65,
  "wind_speed": 12,
  "precipitation": 0,
  "holiday": 0
}

Response:
{
  "prediction": 69.37,
  "message": "PM2.5 prediction for Karachi"
}

2. Health Impact Assessment

POST /health_impact
Request:
{
  "pm25": 120
}

Response:
{
  "pm25_level": 120,
  "aqi_category": "Unhealthy",
  "respiratory_risk": 65.42,
  "cardiovascular_risk": 39.18,
  "health_advice": "Some members of the general public may experience health effects..."
}

🖥️ Frontend Usage

Inputs: Date, temperature, humidity, wind speed, precipitation, holiday status

Outputs:

1. Predicted PM2.5 with color-coded category

2. Health impact assessment

3. Actionable recommendations

📈 Model Performance

| Model             | RMSE | MAE  | R² Score | Training Time |
| ----------------- | ---- | ---- | -------- | ------------- |
| XGBoost           | 4.23 | 3.45 | 0.89     | 12.3s         |
| Random Forest     | 4.56 | 3.78 | 0.87     | 8.7s          |
| Linear Regression | 5.89 | 4.92 | 0.78     | 1.2s          |

Feature Importance (XGBoost):

Temperature (25.4%)

Humidity (18.7%)

Wind Speed (15.2%)

Day of Week (12.1%)

Holiday Status (9.8%)

Precipitation (8.3%)

Seasonal Factors (10.5%)

🔧 Technical Details

Pipeline: Data collection → Feature engineering → Model training → API → Frontend

Algorithms: XGBoost (main), Random Forest, Linear Regression

Validation: TimeSeriesSplit, grid search tuning

API: RESTful with JSON, error handling, input validation

🚀 Deployment

Docker

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]


docker build -t air-quality-api .
docker run -p 5000:5000 air-quality-api

🤝 Contributing

Fork repo → Create branch → Commit → Push → Pull request

Run tests: pytest tests/

Check code style: flake8

Type check: mypy

Contribution areas:

Real-time OpenAQ integration

Additional ML models

Enhanced dashboards

Mobile app

Deployment automation

📄 License

This project is licensed under the MIT License.

👨‍💻 Author

Muhammad Ghous

🙏 Acknowledgments

Pakistan Environmental Protection Agency

OpenAQ platform

Karachi health authorities

Open-source data science community