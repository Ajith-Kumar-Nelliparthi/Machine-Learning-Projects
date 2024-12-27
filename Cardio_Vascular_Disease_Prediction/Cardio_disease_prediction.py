import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
import xgboost  as xgb 



df = pd.read_csv('C:\Machine Learning Projects\Cardio_Disease_Vascular\cardio_train.csv',sep=';')
df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(0.975)) | (df['ap_hi'] < df['ap_hi'].quantile(0.025))].index,inplace=True)
df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(0.975)) | (df['ap_lo'] < df['ap_lo'].quantile(0.025))].index,inplace=True)
df['age'] = (df['age']/365.25).round().astype(int)


df_full_train,df_test = train_test_split(df,test_size=0.2,random_state=42)
df_train,df_val = train_test_split(df_full_train,test_size=0.25,random_state=42)

df_test = df_test.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

x_test = df_test.drop(columns=['cardio'])
x_train = df_train.drop(columns=['cardio'])
x_val = df_val.drop(columns=['cardio'])

y_test = df_test['cardio'].values
y_train = df_train['cardio'].values
y_val = df_val['cardio'].values

del df_test['cardio']
del df_train['cardio']
del df_val['cardio']

features = ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo','cholesterol', 'gluc', 'smoke', 'alco', 'active']

train_dicts = df_train[features].to_dict(orient='records')
val_dicts = df_val[features].to_dict(orient='records')

dv = DictVectorizer(sparse=True)
x_train = dv.fit_transform(train_dicts)
x_val = dv.transform(val_dicts)


import xgboost as xgb
best_params = {'max_depth': 5,
               'learning_rate': 0.1}
xgb = xgb.XGBClassifier(**best_params,random_state=0)
xgb.fit(x_train,y_train)
y_pred = xgb.predict(x_val)
accuracy = accuracy_score(y_val,y_pred)
rmse = np.sqrt(mean_squared_error(y_val,y_pred))
print(f"Test Accuracy: {accuracy}")
print(f"Test RMSE: {rmse}")

import pickle
filename = 'cardio_xgb.pkl'
with open(filename,'wb') as f_out:
    pickle.dump(xgb,f_out)

dv_file = 'cardio_dv.bin'
with open(dv_file,'wb') as f_out:
    pickle.dump(dv,f_out)
print("Model and DictVectorizer are saved to disk")