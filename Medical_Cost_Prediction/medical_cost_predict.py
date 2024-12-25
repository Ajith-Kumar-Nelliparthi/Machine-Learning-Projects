import requests
import numpy as np

url = 'http://localhost:5000/predict'

# Medical Cost Prediction API Client
medical_values = {'age': 19,
 'sex': 'female',
 'bmi': 27.9,
 'children': 0,
 'smoker': 'yes',
 'region': 'southwest'}

response = requests.post(url, json=medical_values).json()

# Print the predicted medical cost
# Assuming 'response' is the dictionary received
predicted_cost = response['predicted_medical_cost']
print(f'Predicted medical cost: {np.expm1(predicted_cost)}')  # Use expm1 to reverse log1p