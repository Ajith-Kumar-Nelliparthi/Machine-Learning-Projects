import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf


# Load the model
try:
    model = tf.keras.models.load_model('lung_cancer_classifier_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

# Load image 
img_path = 'lungn182.jpeg'

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))  # Resize the image to the target size
img_array = image.img_to_array(img)  # Convert the image to a numpy array
img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
img_array /= 255.0  # Normalize the image array to [0, 1]

# Make prediction
prediction = model.predict(img_array)

# Interpret the prediction
predicted_class = np.argmax(prediction, axis=1)  # Get the index of the class with the highest probability
print(f'Predicted class: {predicted_class[0]}')
# Assuming you have already loaded your model and made a prediction
predicted_class_index = predicted_class[0]  # Get the predicted class index

# Define the class names
class_names = ['Lung_adenocarcinoma', 'Lung_benign_tissue', 'Lung_squamous_cell_carcinoma']

# Print the predicted class name
print(f'Predicted class: {class_names[predicted_class_index]}')