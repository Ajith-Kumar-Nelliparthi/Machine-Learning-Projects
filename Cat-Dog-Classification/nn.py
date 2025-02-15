# Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from sklearn.model_selection import train_test_split
import numpy as np


# Load data
X = pickle.load(open("X.pickle",'rb'))
y = pickle.load(open("y.pickle",'rb'))

y = np.array(y, dtype=np.float32)  # Convert labels to NumPy array


# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train , X_test = X_train/255.0 ,X_test/255.0 #normalize pixel value

# reshape data to fit cnn model
X_train = X_train.reshape(-1,80,80,1)
X_test = X_test.reshape(-1,80,80,1)


# build the cnn model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), strides=(2,2), activation='relu', input_shape=X_train.shape[1:]),
    layers.Dropout(0.3),
    layers.Conv2D(64, (3 ,3), strides=(2,2), activation='relu'),
    layers.Dropout(0.4),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.GlobalAveragePooling2D(),                     # Converts the 3D feature maps into a 1D vector
    layers.Dense(64, activation='relu'), 
    layers.Dropout(0.5),   
    layers.Dense(1, activation='sigmoid') 
])

# compile the model
model.compile(optimizer='adam',
              loss = 'binary_crossentropy',     
              metrics=['accuracy'])

print(f"X shape: {X.shape}, dtype: {X.dtype}")
print(f"y shape: {y.shape}, dtype: {y.dtype}, first 5 labels: {y[:5]}")


# train the model
batch_size = 32
history = model.fit(X_train, y_train, epochs=25, batch_size=batch_size,validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nTest accuracy: {test_acc:.4f}')


model.save("pet_classifier_model.h5")  # Saves the model in HDF5 format

print(f"model saved to {'pet_classifier_model.h5'}")

