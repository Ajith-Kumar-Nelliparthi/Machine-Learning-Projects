import requests
import json

url = 'http://localhost:9999/predict'

data = {'cap-shape': 'x',
 'cap-surface': 's',
 'cap-color': 'n',
 'bruises': 't',
 'odor': 'p',
 'gill-attachment': 'f',
 'gill-spacing': 'c',
 'gill-size': 'n',
 'gill-color': 'k',
 'stalk-shape': 'e',
 'stalk-root': 'e',
 'stalk-surface-above-ring': 's',
 'stalk-surface-below-ring': 's',
 'stalk-color-above-ring': 'w',
 'stalk-color-below-ring': 'w',
 'veil-type': 'p',
 'veil-color': 'w',
 'ring-number': 'o',
 'ring-type': 'p',
 'spore-print-color': 'k',
 'population': 's',
 'habitat': 'u'
 }

response = requests.post(url,json=data).json()
print(response)


# Interpret the prediction
if response['prediction'] == 1:
    print("The mushroom is likely to be poisonous.")
else:
    print("The mushroom is likely to be edible.")