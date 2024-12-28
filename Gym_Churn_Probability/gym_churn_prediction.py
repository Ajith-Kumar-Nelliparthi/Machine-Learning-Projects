# Import necessary libraries
import pandas as pd  # Library for data manipulation and analysis
import numpy as np  # Library for numerical operations
import matplotlib.pyplot as plt  # Library for creating static, animated, and interactive visualizations
import seaborn as sns  # Library for creating informative and attractive statistical graphics
from sklearn.model_selection import train_test_split  # Function for splitting data into training and testing sets
from sklearn.feature_extraction import DictVectorizer  # Class for converting dictionaries into numerical features
from sklearn.utils import resample  # Function for resampling data
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error  # Functions for evaluating model performance
from sklearn.ensemble import GradientBoostingClassifier  # Class for gradient boosting classification

# Load the dataset
df = pd.read_csv('gym_churn_us.csv')  # Read the CSV file into a DataFrame

# Convert column names to lowercase
df.columns = df.columns.str.lower()  # Convert column names to lowercase for consistency

# Separate minority and majority classes
minority = df[df['churn'] == 1]  # Get rows where 'churn' is 1 (minority class)
majority = df[df['churn'] == 0]  # Get rows where 'churn' is 0 (majority class)

# Upsample the minority class
minority_upsampled = resample(minority, replace=True, n_samples=2939, random_state=42)  # Upsample the minority class to match the majority class size

# Combine the upsampled minority class with the majority class
df = pd.concat([majority, minority_upsampled])  # Combine the upsampled minority class with the majority class

# Drop unnecessary columns
df = df.drop(['gender', 'phone'], axis=1)  # Drop 'gender' and 'phone' columns

# Split the data into training and testing sets
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)  # Split the data into training and testing sets
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)  # Split the training data into training and validation sets

# Reset the index of the DataFrames
df_test = df_test.reset_index(drop=True)  # Reset the index of the testing DataFrame
df_train = df_train.reset_index(drop=True)  # Reset the index of the training DataFrame
df_val = df_val.reset_index(drop=True)  # Reset the index of the validation DataFrame

# Split the data into features and target
y_train = df_train['churn'].values  # Get the target values for the training data
y_test = df_test['churn'].values  # Get the target values for the testing data
y_val = df_val['churn'].values  # Get the target values for the validation data

x_train = df_train.drop(['churn'], axis=1)  # Get the features for the training data
x_test = df_test.drop(['churn'], axis=1)  # Get the features for the testing data
x_val = df_val.drop(['churn'], axis=1)  # Get the features for the validation data

# Delete the target column from the DataFrames
del df_train['churn']  # Delete the target column from the training DataFrame
del df_test['churn']  # Delete the target column from the testing DataFrame
del df_val['churn']  # Delete the target column from the validation DataFrame

# Convert the DataFrames to dictionaries
train_dicts = df_train.to_dict(orient='records')  # Convert the training DataFrame to a list of dictionaries
val_dicts = df_val.to_dict(orient='records')  # Convert the validation DataFrame to a list of dictionaries

# Create a DictVectorizer object
dv = DictVectorizer(sparse=True)  # Create a DictVectorizer object to convert dictionaries into numerical features

# Fit the DictVectorizer object to the training data and transform both the training and validation data
x_train = dv.fit_transform(train_dicts)  # Fit the DictVectorizer object to the training data and transform the training data
x_val = dv.transform(val_dicts)  # Transform the validation data using the DictVectorizer object

# Define the hyperparameters for the Gradient Boosting Classifier
best_params = {'n_estimators': 500,  # Number of estimators
               'min_samples_split': 2,  # Minimum number of samples required to split an internal node
               'min_samples_leaf': 1,  # Minimum number of samples required to be at a leaf node
               'max_features': 'sqrt',  # Maximum number of features to consider at each split
               'max_depth': 10,  # Maximum depth of the tree
               'learning_rate': 0.1}  # Learning rate of the classifier

# Create a Gradient Boosting Classifier object
gbm = GradientBoostingClassifier(random_state=42, verbose=1)  # Create a Gradient Boosting Classifier object with random state and verbosity

# Set the hyperparameters of the Gradient Boosting Classifier
gbm.set_params(**best_params)  # Set the hyperparameters of the Gradient Boosting Classifier

# Train the Gradient Boosting Classifier
gbm.fit(x_train, y_train)  # Train the Gradient Boosting Classifier using the training data

# Make predictions using the Gradient Boosting Classifier
y_pred = gbm.predict(x_val)  # Make predictions using the Gradient Boosting Classifier

# Evaluate the performance of the Gradient Boosting Classifier
accuracy = accuracy_score(y_val, y_pred)  # Calculate the accuracy of the classifier
roc_auc = roc_auc_score(y_val, y_pred)  # Calculate the ROC AUC score of the classifier
mean_squared = mean_squared_error(y_val, y_pred)  # Calculate the mean squared error of the classifier

# Print the performance metrics
print(f'Accuracy: {accuracy}')  # Print the accuracy of the classifier
print(f'ROC AUC: {roc_auc}')  # Print the ROC AUC score of the classifier
print(f'Mean Squared Error: {mean_squared}')  # Print the mean squared error of the classifier

import pickle  # Library for serializing and deserializing Python objects
# Save the trained model to a file
filename = 'gym_churn_model.pkl'  # Define the filename for the trained model
with open('gym_churn_model.pkl', 'wb') as f_out:  # Open the file in binary write mode
    pickle.dump(gbm, f_out)  # Save the trained model to the file

# Save the DictVectorizer object to a file
dv_filename = 'gym_churn_dv.pkl'  # Define the filename for the DictVectorizer object
with open(dv_filename, 'wb') as f_out:  # Open the file in binary write mode
    pickle.dump(dv, f_out)  # Save the DictVectorizer object to the file

print(f"the file is saved to {filename} and {dv_filename}")  # Print the filename of the saved model and DictVectorizer object