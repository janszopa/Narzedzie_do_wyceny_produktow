import pandas as pd
import numpy as np

from joblib import load
from sklearn.preprocessing import StandardScaler

model = load('model/model.joblib')
model_columns = load('model/columns.joblib')
scaler = load('model/scaler.joblib')

data = np.array([[0.23, 61.5, 55.0, 3.95, 3.98, 2.43, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
data = scaler.transform(data)

#df = pd.get_dummies(data, columns=['cut', 'color', 'clarity'])

pred = model.predict(data)
print(pred)