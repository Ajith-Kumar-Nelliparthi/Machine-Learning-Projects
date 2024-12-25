import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_csv('C:\Machine Learning Projects\Medical_Cost_Prediction\expenses.csv')
cat_variables = df.select_dtypes(exclude='number').columns
num_variables = df.select_dtypes(exclude='object').columns 

num_cols = len(df.columns)
cols_per_row = 2
num_rows = (num_cols + cols_per_row - 1) // cols_per_row
plt.figure(figsize=(15,5 * num_rows))
for i,col in enumerate(df.columns,1): 
    plt.subplot(num_rows,cols_per_row,i) 
    sns.histplot(data=df,x=col)
    plt.title(f'Histplot of {col}')
plt.tight_layout()
plt.show()


x = df.drop(columns=['charges'])
y = df['charges']
sns.pairplot(df,y_vars=['charges'],x_vars=x.columns)


df_full_train,df_test = train_test_split(df,test_size=0.2,random_state=42)
df_train,df_val = train_test_split(df_full_train,test_size=0.25,random_state=42) 
df_test = df_test.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = np.log1p(df_train.charges.values)
y_test = np.log1p(df_test.charges.values)
y_val = np.log1p(df_val.charges.values)

x_train = df_train.drop(columns=['charges'])
x_test = df_test.drop(columns=['charges'])
x_val = df_val.drop(columns=['charges'])

features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

train_dicts = df_train[features].to_dict(orient='records')
val_dicts = df_val[features].to_dict(orient='records')

dv = DictVectorizer(sparse=True)
x_train = dv.fit_transform(train_dicts)
x_val = dv.transform(val_dicts)

rf = RandomForestRegressor(n_estimators=10,
                          max_depth=5,
                          random_state=42,
                          n_jobs=-1)
rf.fit(x_train,y_train)
y_pred = rf.predict(x_val)
mse = mean_squared_error(y_val,y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error {rmse}')

import pickle
filename = 'medical_cost.pkl'
with open(filename,'wb') as f_out:
    pickle.dump(rf,f_out)

dv_filename = 'medical_dv.pkl'
with open(dv_filename, 'wb') as file:  # Use 'wb' for writing
    pickle.dump(dv, file)
print(f'the o/p file is saved to {filename}')