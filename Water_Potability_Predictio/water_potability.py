# Water Potability Prediction Guide
# ======================================

# Table of Contents
# -----------------

# 1. [Introduction](#introduction)
# 2. [Importing Libraries](#importing-libraries)
# 3. [Loading Data](#loading-data)
# 4. [Data Preprocessing](#data-preprocessing)
# 5. [Feature Engineering](#feature-engineering)
# 6. [Model Training](#model-training)
# 7. [Model Evaluation](#model-evaluation)
# 8. [Model Deployment](#model-deployment)

# Introduction
# ------------

# This guide provides a step-by-step walkthrough of building a water potability prediction model using Python.

# Importing Libraries
# Import necessary libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import export_text
import zipfile

# Loading Data
# Load the water potability dataset
df = pd.read_csv('water_potability.csv')

# Data Preprocessing
# Convert column names to lowercase
df.columns = df.columns.str.lower()

# Handle missing values
null_cols = ['ph','sulfate','trihalomethanes']
for col in null_cols: 
    df[col] = df[col].fillna(df[col].mean())

# Split data into training and testing sets
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# Reset indices
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

# Prepare target variables
y_train = np.log1p(df_train.potability.values)
y_test = np.log1p(df_test.potability.values)
y_val = np.log1p(df_val.potability.values)

# Drop the target variable from the feature DataFrames
x_train = df_train.drop(columns=['potability'])
x_test = df_test.drop(columns=['potability'])
x_val = df_val.drop(columns=['potability'])

# Feature Engineering
# Select relevant features
features = ['ph', 'hardness', 'solids', 'chloramines', 'sulfate', 'conductivity','organic_carbon', 'trihalomethanes', 'turbidity']

# Convert data to dictionaries
train_dicts = df_train[features].to_dict(orient='records')
val_dicts  = df_val[features].to_dict(orient='records')

# Use DictVectorizer to convert dictionaries to sparse matrices
dv = DictVectorizer(sparse=True)
x_train = dv.fit_transform(train_dicts)
x_val = dv.fit_transform(val_dicts)

# Model Training
# Train a random forest regressor model
rf = RandomForestRegressor(n_estimators=50,
                           max_depth=25,
                           min_samples_leaf=4,
                           min_samples_split=10,
                           max_features='sqrt',
                           random_state=1,
                           n_jobs=-1) 
rf.fit(x_train,y_train)

# Model Evaluation
# Make predictions on the validation set
y_pred = rf.predict(x_val)

# Calculate the mean squared error
mse = mean_squared_error(y_val,y_pred)
rmse = np.sqrt(mse)
print(f'root mean squared error {rmse}')

# Model Deployment
# Save the trained model to a file
import pickle
filename = "water_potability.pkl"
with open(filename, 'wb') as f_out:
    pickle.dump(rf,f_out)

dv_filename = 'water_dv.pkl'
with open(dv_filename, 'wb') as file:  # Use 'wb' for writing
    pickle.dump(dv, file)
print(f'the o/p file is saved to {filename}')