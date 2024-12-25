from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import json
import pickle

model_file = 'medical_cost.pkl'
with open(model_file,'rb') as f_in:
    rf = pickle.load(f_in)

dv_filename = 'medical_dv.pkl'
with open(dv_filename,'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('Medical_Cost_Prediction')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        medical_values = request.get_json()
        if medical_values is None:
            return jsonify({"error": "No input data provided"}), 400
        x = pd.DataFrame([medical_values])
        x = dv.transform([medical_values])
        prediction = rf.predict(x)
        result = {
            'predicted_medical_cost': float(prediction[0])
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0' ,port=5000)