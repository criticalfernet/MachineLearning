import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Regression import RegressionModel

data = pd.read_csv('non-linear.csv')
x = data[['x']].values
y = data['y'].values

model = RegressionModel(x,y,15,True)

theta = model.train(0.01,100000)

