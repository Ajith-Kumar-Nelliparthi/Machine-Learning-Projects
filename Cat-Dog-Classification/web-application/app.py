# import libraries
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import os 
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# load the trained model
model = load_model("pet_classifier_model.h5") 

# preprocess the image
def preprocess_image(img):
    img = img.convert("L") # convert to grayscale
    img = img.resize((80,80)) # resize to 80x80
    img_arr = np.array(img) / 255.0 # normalize pixel values to [0,1]
    img_arr = img_arr.reshape(1,80,80,1) # Reshape for model input
    return img_arr

# route for frontend
@app.route('/')
def home():
    return render_template('index.html')

# api for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file Uploaded"}), 400
    
    file = request.files['file']
    img = Image.open(file)

    # preprocess and predict
    img_arr = preprocess_image(img)
    prediction = model.predict(img_arr)[0][0]

    # convert probability to class
    label = 'Cat' if prediction > 0.5 else "Dog"

    return jsonify({'prediction': float(prediction), "class":label})

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=8000)