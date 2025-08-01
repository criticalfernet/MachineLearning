import numpy as np
import pandas as pd
from Regression import RegressionModel

data = pd.read_csv('2017.csv')
x_data = data[['Economy..GDP.per.Capita.']]
y_data = data['Happiness.Score']

print(x_data.shape)
model = RegressionModel(x_data,y_data)

w = model.train(0.01,5000)

