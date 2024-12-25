from flask import Flask,request,jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model_file = 'fetal_health_predict.pkl'
with open(model_file, 'rb') as f_in:
    gbm = pickle.load(f_in)

dv_filename = 'fetal_health_dv.pkl'
with open(dv_filename, 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('fetal_health')

@app.route('/predict',methods=['POST'])
def predict():
    fetal_data = request.get_json()
    if fetal_data is None:
        return jsonify({"error": "No input data provided"}), 400
    x = pd.DataFrame([fetal_data])
    x = dv.transform([fetal_data])
    y_pred_proba = gbm.predict_proba(x)[0]

    # Get the predicted class (most likely class)
    y_pred_class = gbm.predict(x)[0]

    # Get the specific probability for class 1 (assuming 1 is the class of interest)
    y_pred_specific = float(y_pred_proba[1])  # Probability of class 1

    # Logic for classification decision (custom thresholds can be set here)
    if y_pred_class == 0:  # If the most likely class is 0
        health_classification = 'normal'
    elif y_pred_class == 1:  # If the most likely class is 1
        health_classification = 'suspect'
    elif y_pred_class == 2:  # If the most likely class is 2
        health_classification = 'pathological'

    result = {
        'health_classification': health_classification,
        'probabilities': {
            'normal': float(y_pred_proba[0]),
            'suspect': float(y_pred_proba[1]),
            'pathological': float(y_pred_proba[2])
        },
        'specific_class_probability': y_pred_specific
    }

    return jsonify(result)
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=5000)