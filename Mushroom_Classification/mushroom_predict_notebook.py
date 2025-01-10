import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score,RandomizedSearchCV,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error,classification_report,confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('mushrooms.csv')
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

data_dict = df.to_dict(orient='records')
x = [i for i in data_dict]
y = [i.pop('class') for i in data_dict]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
y_train = np.array(y_train)

dv = DictVectorizer(sparse=False)
train_dicts = dv.fit_transform(x_train)
val_dicts = dv.transform(x_test)

model = RandomForestClassifier(n_estimators=100,random_state=42)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)
cv_scores = []
for train_index,test_index in skf.split(train_dicts,y_train):
    x_train_fold,x_test_fold = train_dicts[train_index],train_dicts[test_index]
    y_train_fold,y_test_fold = y_train[train_index],y_train[test_index]
    model.fit(x_train_fold,y_train_fold)
    y_pred_fold = model.predict(x_test_fold)
    acc = accuracy_score(y_test_fold,y_pred_fold)
    cv_scores.append(acc)
print(f'Cross validation accuracy: {np.mean(cv_scores) * 100:.2f}%')

import pickle

filename = 'mushroom_model.pkl'
with open('mushroom_model.pkl','wb') as f_out:
    pickle.dump(model,f_out)

dv_filename = 'mushroom_dv.pkl'
with open(dv_filename,'wb') as f_out:
    pickle.dump(dv,f_out)

print(f"the file is saved to {filename} and {dv_filename}")  