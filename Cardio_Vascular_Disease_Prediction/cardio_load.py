from flask import Flask, request, jsonify
import pandas as pd 
import numpy as np
import pickle

# Load the model and vectorizer
model_file = 'cardio_xgb.pkl'
with open('cardio_xgb.pkl', 'rb') as f_in:
    xgb = pickle.load(f_in)

dv_filename = 'cardio_dv.bin'
with open(dv_filename, 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask(__name__)
@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()
    x = pd.DataFrame([data])
    x = dv.transform([data])
    y_pred = xgb.predict(x)

    result = {
            'disease_probability': float(y_pred[0])
        }
        
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9999)

