import requests

# URL of the prediction endpoint
url = 'http://localhost:9999/predict'  # Make sure this matches the port your Flask app is running on

# Patient data
patient_data = {
  'id': 989,
 'age': 68,
 'gender': 1,
 'height': 155,
 'weight': 69.0,
 'ap_hi': 130,
 'ap_lo': 80,
 'cholesterol': 2,
 'gluc': 1,
 'smoke': 1,
 'alco': 1,
 'active': 1}

response = requests.post(url, json=patient_data).json()
predicted_disease_probability = response.get('disease_probability')
print("predicted_disease_probability:",predicted_disease_probability)


threshold = 0.5
if predicted_disease_probability > threshold: 
    print("The patient is predicted to have cardiovascular disease.")
else:
    print("The patient is predicted not to have cardiovascular disease.")