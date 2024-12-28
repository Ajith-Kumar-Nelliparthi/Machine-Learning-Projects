from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the model
model_file = 'gym_churn_model.pkl'
with open(model_file, 'rb') as f_in:
    gbm = pickle.load(f_in)

# Load the DictVectorizer
dv_file = 'gym_churn_dv.pkl'
with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Transform data
        x = dv.transform([data])

        # Predict
        y_pred = gbm.predict(x)

        # Return prediction
        return jsonify({'prediction': y_pred.tolist()[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
