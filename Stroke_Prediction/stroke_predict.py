import requests
url = 'http://localhost:5000/predict'


person_id = 'abc123'
sample_person = {'id': 68023,
 'gender': 'Male',
 'age': 79.0,
 'hypertension': 0,
 'heart_disease': 0,
 'ever_married': 'Yes',
 'work_type': 'Private',
 'Residence_type': 'Rural',
 'avg_glucose_level': 72.73,
 'bmi': 28.4,
 'smoking_status': 'never smoked'}

response = requests.post(url, json=sample_person).json()

# Extract the predicted stroke probability
predicted_stroke_probability = response.get('stroke_probability')

# Check if the prediction is valid
if predicted_stroke_probability is None:
    print("Error: The response does not contain 'stroke_probability'")
else:
    # Assuming a threshold of 0.5 for determining stroke risk
    if predicted_stroke_probability >= 0.5:  # Adjust threshold as needed
        print(f'Person will have a stroke: {person_id}')
    else:
        print(f'Person will not have a stroke: {person_id}')