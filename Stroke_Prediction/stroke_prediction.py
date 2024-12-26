import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import accuracy_score,mean_squared_error,confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier


df = pd.read_csv('C:\Machine Learning Projects\Stroke_Prediction\healthcare-dataset-stroke-data.csv')
df['bmi'] = df['bmi'].replace(np.nan, df['bmi'].mean())
cat_cols = df.select_dtypes(exclude='number')
num_cols = df.select_dtypes(exclude='object')
df['gender'] = df['gender'].replace('Other', df['gender'].mode()[0])


df_minority = df[df['stroke']==1]
df_majority = df[df['stroke'] == 0]
df_minority_upsampled = resample(df_minority,replace=True,
        n_samples=len(df_majority),
        random_state=42)
df_upsampled = pd.concat([df_majority,df_minority_upsampled])
df_upsampled.reset_index(drop=True)


df_full_train,df_test = train_test_split(df_upsampled,test_size=0.2,random_state=1)
df_train,df_val = train_test_split(df_full_train,test_size=0.25,random_state=1)

df_test = df_test.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

x_test = df_test.drop(columns=['stroke'])
x_train = df_train.drop(columns=['stroke'])
x_val = df_val.drop(columns=['stroke'])

y_test = df_test['stroke'].values
y_train = df_train['stroke'].values
y_val = df_val['stroke'].values

del df_train['stroke']
del df_val['stroke']
del df_test['stroke']

features = ['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type', 'Residence_type', 
            'avg_glucose_level', 'bmi','smoking_status']

train_dicts = df_train[features].to_dict(orient='records')
val_dicts = df_val[features].to_dict(orient='records')

dv = DictVectorizer(sparse=True)
x_train = dv.fit_transform(train_dicts)
x_val = dv.transform(val_dicts)


brf = BalancedRandomForestClassifier(max_depth=20,n_estimators=170,n_jobs=-1,random_state=42)
brf.fit(x_train,y_train)
y_pred = brf.predict(x_val)


# Calculate metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

import pickle
filename = 'stroke_predict.pkl'
with open(filename,'wb') as f_out:
    pickle.dump(brf,f_out)

dv_filename = 'stroke_health_dv.pkl'
with open(dv_filename,'wb') as file:
    pickle.dump(dv,file)

print(f'model saved to {filename}')