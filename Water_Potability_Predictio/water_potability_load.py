import pickle
from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd


model_file = 'water_potability.pkl'

with open(model_file, 'rb') as f_in:
    rf = pickle.load(f_in)

dv_filename = 'water_dv.pkl'
with open(dv_filename, 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('potability')

@app.route('/')
def home():
    return "Welcome to the Water Potability App!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        water_sample = request.get_json()
        if water_sample is None:
            return jsonify({"error": "No input data provided"}), 400
        
        # Validate input data here (check for required fields, types, etc.)
        
        X = pd.DataFrame([water_sample]) 
        x = dv.transform([water_sample])
        y_pred = rf.predict(x) 
        threshold = 0.5
        potability = y_pred[0] >= threshold

        result = {
            'potability_probability': float(y_pred[0]),  # Ensure to access the first element
            'potability': bool(potability)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

