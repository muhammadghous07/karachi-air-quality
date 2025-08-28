import streamlit as st
import requests
import json
import datetime

# Set up the Streamlit UI
st.title("Karachi Air Quality Prediction")

st.header("Enter Data for Prediction")

# Input widgets
date_input = st.date_input("Select Date", datetime.date.today())
temperature = st.slider("Temperature (°C)", -5, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 10)
precipitation = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0)
holiday = st.selectbox("Is it a holiday?", [0, 1])

# Create a dictionary of the input data
input_data = {
    'date': str(date_input),
    'temperature': temperature,
    'humidity': humidity,
    'wind_speed': wind_speed,
    'precipitation': precipitation,
    'holiday': holiday
}

if st.button("Get Prediction"):
    # The URL to your Flask API. This assumes the API is running on localhost:5000.
    api_url = "http://localhost:5000/predict"

    # Send the data to the API
    try:
        response = requests.post(api_url, json=input_data)
        
        if response.status_code == 200:
            result = response.json()
            predicted_pm25 = result.get("prediction")  # Corrected key from "predicted_pm25" to "prediction"
            
            # Check if predicted_pm25 is valid before formatting
            if predicted_pm25 is not None:
                st.success(f"Predicted PM2.5 Level: {predicted_pm25:.2f} µg/m³")

                # Request health impact
                health_impact_url = "http://localhost:5000/health_impact"
                health_response = requests.post(health_impact_url, json={'pm25': predicted_pm25})
                
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    st.subheader(f"Air Quality: {health_data['aqi_category']}")
                    st.info(health_data['health_advice'])
                else:
                    st.warning("Could not get health impact data from API.")
            else:
                st.error("Prediction value not found in API response.")
                st.write(result)
        else:
            st.error("Error from API. Please check your input.")
            st.write(response.json())

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the API. Error: {e}")