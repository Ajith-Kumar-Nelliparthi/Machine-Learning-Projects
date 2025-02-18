from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import os 
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model('lung_cancer_classifier_model.h5')

# Preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize the image to the target size
    img_arr = np.array(img) / 255.0  # Normalize the image array to [0, 1]
    img_arr = np.expand_dims(img_arr, axis=0)  # Add a batch dimension
    return img_arr

# Route for frontend
@app.route('/')
def home():
    return render_template('index.html')

# API for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img = Image.open(file)

    # Preprocess and predict
    img_arr = preprocess_image(img)
    prediction = model.predict(img_arr)

    # Interpret the prediction
    predicted_class = np.argmax(prediction, axis=1)  # Get the index of the class with the highest probability

    # Get the predicted class index
    predicted_class_index = predicted_class[0]

    # Define the class names
    class_names = ['Lung_adenocarcinoma', 'Lung_benign_tissue', 'Lung_squamous_cell_carcinoma']

    # Return the prediction and the corresponding class name
    return jsonify({'prediction': float(predicted_class_index), 'class': class_names[predicted_class_index]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)