#from PySide6.QtWidgets
import numpy as np
import pandas as pd
import sklearn as sk

from joblib import load




model = load('model/model.joblib')

data1 = pd.read_csv("https://raw.githubusercontent.com/janszopa/Narzedzie_do_wyceny_produktow/main/diamonds.csv")
data = pd.DataFrame(data1)
data= data.drop(columns=['Unnamed: 0', 'price'])
data_test = data.head(10)
print(data_test)

cols_to_encode = data_test.columns[[1, 2, 3]]
dataToPredConverted = pd.get_dummies(data_test, columns=data_test.columns[[1, 2, 3]])

#dataToPredConverted = pd.get_dummies(data_test, columns=['cut', 'color', 'clarity'])
#pred = model.predict(dataToPredConverted)
print(dataToPredConverted)
