from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("pet_classifier_model.h5") 

# Check if the file exists
image_path = r"Dog\130.jpg"             

# Load and preprocess the image
img = Image.open(image_path).convert("L")       # Convert to grayscale
img = img.resize((80, 80))                      # Resize to match model input shape
img_array = np.array(img) / 255.0               # Normalize pixel values
img_array = img_array.reshape(1, 80, 80, 1)     # Reshape for model input

# Make a prediction
predict = model.predict(img_array)

# Convert to binary class (0 or 1)
prediction = int(predict[0][0] > 0.5)

# Print results
print(f"Raw Prediction: {predict[0][0]}")
print(f"Predicted Class: {'Cat' if prediction == 1 else 'Dog'}")
