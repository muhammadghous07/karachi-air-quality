
FROM python:3.9-slim

WORKDIR /app

# Copy requirements file from the 'api' folder and install them
COPY api/requirements.txt .
RUN pip install -r requirements.txt

# Copy all the necessary project files
COPY . .

# Create directory for models
RUN mkdir -p models

# Copy model file from the 'models' folder
COPY models/xgboost_model.pkl models/

# Expose the correct port
EXPOSE 5000

# Run the app.py file which is inside the 'api' folder
CMD ["python", "api/app.py"]
