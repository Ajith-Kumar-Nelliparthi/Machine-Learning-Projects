import requests

url = 'http://localhost:5000/predict'

customer = {'near_location': 1,
 'partner': 0,
 'promo_friends': 0,
 'contract_period': 1,
 'group_visits': 1,
 'age': 29,
 'avg_additional_charges_total': 264.60551415647086,
 'month_to_end_contract': 1.0,
 'lifetime': 0,
 'avg_class_frequency_total': 2.355562612407676,
 'avg_class_frequency_current_month': 2.3723301792604268
 }

response = requests.post(url, json=customer).json()
print("churn probability: ", response)

if response['prediction'] == 0:
    print("Customer will not churn")
else:
    print("Customer will churn")
