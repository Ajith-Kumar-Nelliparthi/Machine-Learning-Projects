import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load model and DV with error handling
try:
    model_file = 'random_forest_model.pkl'
    with open(model_file, 'rb') as f_in:
        model = pickle.load(f_in)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None  # Set model to None if loading fails

try:
    dv_file = 'dv.pkl'
    with open(dv_file, 'rb') as f_in:
        dv = pickle.load(f_in)
except Exception as e:
    logging.error(f"Error loading DV: {e}")
    dv = None  # Set dv to None if loading fails

app = Flask(__name__)
CORS(app)  # Apply CORS to the app

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/styles', methods=['GET'])
def styles():
    return render_template('styles.css')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or dv is None:
        return jsonify({'error': 'Model or DV not loaded'}), 500

    data = request.get_json()
    logging.debug(f'Incoming data: {data}')  # Log incoming data

    try:
        x = pd.DataFrame([data])
        x_transformed = dv.transform(x.to_dict(orient='records'))
        logging.debug(f'Transformed data: {x_transformed}')  # Log transformed data
        y_pred = model.predict(x_transformed)
        return jsonify({'prediction': y_pred.tolist()[0]})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)
