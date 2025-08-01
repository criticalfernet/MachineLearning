import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Regression import RegressionModel

data = pd.read_csv('2017.csv')
inputs = ['Economy..GDP.per.Capita.','Freedom','Health..Life.Expectancy.']
output = 'Happiness.Score'

x_data = data[inputs]
y_data = data[output]

model = RegressionModel(x_data,y_data,0,True)

w = model.train(0.01,5000)

