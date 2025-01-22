import pandas as pd
import numpy as np
import joblib

X1 = {"brand": "BMW", "year": 2020, "engine_size": 4.2, "fuel_type": "Electric",
      "transmission": "Automatic", "mileage": 141294, "condition": "Like New", "model": "X5"}
X2 = {"brand": "Mercedes", "year": 2018, "engine_size": 3.9, "fuel_type": "Hybrid",
      "transmission": "Automatic", "mileage": 218811, "condition": "Like New", "model": "C-Class"}

pipeline_load = joblib.load("pipeline.joblib")
prediction = pipeline_load.predict(pd.DataFrame([X1]))
print(X1, prediction[0])
prediction = pipeline_load.predict(pd.DataFrame([X2]))
print(X2, prediction[0])