import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,mean_squared_error,accuracy_score,classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV


df = pd.read_csv('fetalhealth.csv')
df['fetal_health'] = df['fetal_health'].astype(int)
fetal_health = {
    1:'normal',
    2:'suspect',
    3:'pathological'
}
df.fetal_health = df.fetal_health.map(fetal_health)

num_cols = len(df.columns)  # Get the number of columns
cols_per_row = 2  # Define how many plots per row
num_rows = (num_cols + cols_per_row - 1) // cols_per_row  # Calculate number of rows needed
plt.figure(figsize=(15, 5 * num_rows))  # Adjust the figure size based on the number of rows
for i, col in enumerate(df.columns, 1):
    plt.subplot(num_rows, cols_per_row, i)  # Create a subplot for each column
    sns.histplot(data=df, x=col)
    plt.title(f'Histplot of {col}')
plt.tight_layout()
plt.show()

num_cols = len(df.columns)  # Get the number of columns
cols_per_row = 2  # Define how many plots per row
num_rows = (num_cols + cols_per_row - 1) // cols_per_row  # Calculate number of rows needed
plt.figure(figsize=(15, 5 * num_rows))  # Adjust the figure size based on the number of rows
# Loop through the columns and create a subplot for each
for i, col in enumerate(df.columns, 1):
    plt.subplot(num_rows, cols_per_row, i)  # Create a subplot for each column
    sns.boxplot(data=df, x=col)
    plt.title(f'Histplot of {col}')
# Adjust layout and display the plot
plt.tight_layout()
plt.show()

df_full_train,df_test = train_test_split(df,test_size=0.2,random_state=0) 
df_train,df_val = train_test_split(df_full_train,test_size=0.25,random_state=0) 

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = df_train['fetal_health'].values
y_test = df_test['fetal_health'].values
y_val = df_val['fetal_health'].values

x_train = df_train.drop(columns=['fetal_health'])
x_test = df_test.drop(columns=['fetal_health'])
x_val = df_val.drop(columns=['fetal_health'])

features = ['baseline value', 'accelerations', 'fetal_movement',
       'uterine_contractions', 'light_decelerations', 'severe_decelerations',
       'prolongued_decelerations', 'abnormal_short_term_variability',
       'mean_value_of_short_term_variability',
       'percentage_of_time_with_abnormal_long_term_variability',
       'mean_value_of_long_term_variability', 'histogram_width',
       'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
       'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
       'histogram_median', 'histogram_variance', 'histogram_tendency']

train_dicts = df_train[features].to_dict(orient='records')
val_dicts = df_val[features].to_dict(orient='records')

dv = DictVectorizer(sparse=True)
x_train = dv.fit_transform(train_dicts)
x_val = dv.transform(val_dicts)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)



best_params = {'n_estimators': 190,
  'min_samples_split': 190,
  'min_samples_leaf': 30,
  'max_depth': 5}

gbm = GradientBoostingClassifier(**best_params,learning_rate=0.1,random_state=0) 
gbm.fit(x_train,y_train)
y_pred = gbm.predict(x_val)
accuracy = accuracy_score(y_val,y_pred)
rmse = np.sqrt(mean_squared_error(y_val,y_pred))
print(f"Test Accuracy: {accuracy}")
print(f"Test RMSE: {rmse}")

import pickle
filename = 'fetal_health_predict.pkl'
with open('fetal_health_predict.pkl','wb') as f_out:
    pickle.dump(gbm,f_out)

dv_filename = 'fetal_health_dv.pkl'
with open(dv_filename,'wb') as file:
    pickle.dump(dv,file)

print(f'model saved to {filename}')