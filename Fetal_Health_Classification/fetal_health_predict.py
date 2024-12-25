import requests
url = 'http://localhost:5000/predict'

fetal_data = {
    'baseline value': 120.0,
    'accelerations': 0.0,
    'fetal_movement': 0.0,
    'uterine_contractions': 0.0,
    'light_decelerations': 0.0,
    'severe_decelerations': 0.0,
    'prolongued_decelerations': 0.0,
    'abnormal_short_term_variability': 73.0,
    'mean_value_of_short_term_variability': 0.5,
    'percentage_of_time_with_abnormal_long_term_variability': 43.0,
    'mean_value_of_long_term_variability': 2.4,
    'histogram_width': 125.0,
    'histogram_min': 53.0,
    'histogram_max': 178.0,
    'histogram_number_of_peaks': 8.0,
    'histogram_number_of_zeroes': 0.0,
    'histogram_mode': 143.0,
    'histogram_mean': 128.0,
    'histogram_median': 137.0,
    'histogram_variance': 65.0,
    'histogram_tendency': 1.0
}


response = requests.post(url, json=fetal_data).json()


# Handle the response to generate the message based on the classification
health_classification = response['health_classification']
probabilities = response['probabilities']
max_class_label = health_classification  # This is what you already got from the response
max_class_prob = probabilities[max_class_label]

# Custom message based on the classification
if max_class_label == 'pathological':
    custom_message = f"The fetus needs more care. The probability of being pathological is {max_class_prob*100:.2f}%."
elif max_class_label == 'normal':
    custom_message = f"The fetus is healthy. The probability of being normal is {max_class_prob*100:.2f}%. Keep monitoring."
else:  # suspect
    custom_message = f"Alarm! The fetus is in a suspect state. Probability of being suspect is {max_class_prob*100:.2f}%. Caution is needed."

# Print the response and the custom message
print(response)
print("Custom Message:", custom_message)