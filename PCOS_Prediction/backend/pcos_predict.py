import requests
import json

url = 'http://localhost:8000/predict'

# Function to collect user input with validation
def get_user_input():
    data = {}
    try:
        data['Country'] = float(input("Enter Country (numeric value): "))
        data['Age'] = float(input("Enter Age: "))
        data['Menstrual Regularity'] = float(input("Enter Menstrual Regularity (0 or 1): "))
        data['Hirsutism'] = float(input("Enter Hirsutism (0 or 1): "))
        data['Acne Severity'] = float(input("Enter Acne Severity (0 or 1): "))
        data['Family History of PCOS'] = float(input("Enter Family History of PCOS (0 or 1): "))
        data['Insulin Resistance'] = float(input("Enter Insulin Resistance (0 or 1): "))
        data['Lifestyle Score'] = float(input("Enter Lifestyle Score (0 to 1): "))
        data['Stress Levels'] = float(input("Enter Stress Levels (0 to 1): "))
        data['Urban/Rural'] = float(input("Enter Urban/Rural (0 or 1): "))
        data['Awareness of PCOS'] = float(input("Enter Awareness of PCOS (0 or 1): "))
        data['Fertility Concerns'] = float(input("Enter Fertility Concerns (0 or 1): "))
        data['BMI_Obese'] = float(input("Enter BMI Obese (0 or 1): "))
        data['BMI_Overweight'] = float(input("Enter BMI Overweight (0 or 1): "))
        data['BMI_Underweight'] = float(input("Enter BMI Underweight (0 or 1): "))
        data['Socioeconomic Status_Low'] = float(input("Enter Socioeconomic Status Low (0 or 1): "))
        data['Socioeconomic Status_Middle'] = float(input("Enter Socioeconomic Status Middle (0 or 1): "))
        data['Ethnicity_Asian'] = float(input("Enter Ethnicity Asian (0 or 1): "))
        data['Ethnicity_Caucasian'] = float(input("Enter Ethnicity Caucasian (0 or 1): "))
        data['Ethnicity_Hispanic'] = float(input("Enter Ethnicity Hispanic (0 or 1): "))
        data['Ethnicity_Other'] = float(input("Enter Ethnicity Other (0 or 1): "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return None
    return data

# Collect user input
data = get_user_input()
if data is None:
    exit()

# Send request to the backend
try:
    response = requests.post(url, json=data).json()
    if 'prediction' in response:
        print('You are likely to have PCOS' if response['prediction'] == 1 else 'You are not likely to have PCOS')
    else:
        print("Unexpected response from the server.")
except Exception as e:
    print(f"Error communicating with the backend: {e}")
