import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict(x,w,b):
    return np.dot(x,w) + b

def cost_function(x,w,b,y):
    m = len(y)
    y_hat = predict(x,w,b)
    cost = (1 / (2 * m)) * np.sum((y_hat - y) ** 2)
    return cost

def find_gradient(x,y,w,b):
    m = len(y)
    y_hat = predict(x,w,b)
    error = y_hat - y
    dw = (1/m)*np.dot(x.T,error)
    db = (1/m)*np.sum(error)
    return dw, db

def gradient_descend(x,y,w1,b1,alpha,n_iter):
    w = w1
    b = b1
    cost_history = []

    for i in range(n_iter):
        dw, db = find_gradient(x,y,w,b)
        w -= alpha*dw
        b -= alpha*db

        cost = cost_function(x,w,b,y)
        cost_history.append(cost)

    return w,b,cost_history

data: pd.DataFrame = pd.read_csv('2017.csv')
train_data: pd.DataFrame = data.sample(frac=0.8)
test_data: pd.DataFrame = data.drop(train_data.index)

inputs = ['Economy..GDP.per.Capita.','Freedom','Health..Life.Expectancy.']
output = 'Happiness.Score'

x_train = train_data[inputs].values
y_train = train_data[output].values

x_test = test_data[inputs].values
y_test = test_data[output].values

w = np.array([0.0, 0.0,0.0])
b = 0.0 

print("Initial Cost: {:.2f}".format(cost_function(x_train,w,b,y_train)))
w,b,cost_h = gradient_descend(x_train,y_train,w,b,0.01,500)
print("Final Cost: {:.2f}".format(cost_function(x_train,w,b,y_train)))

plt.plot(range(500), cost_h)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Progress')
plt.show()

#cost = cost_function(x_train, y_train, w, b)
#print('Initial cost: {:.2f}'.format(cost))
#
#w,b = gradient_descend(x_train,y_train,w,b,0.01,50000)
#
#cost = cost_function(x_train, y_train, w, b)
#print('Optimized cost: {:.2f}'.format(cost))
#
#x_predictions = np.linspace(x_train.min(), x_train.max(), 100)
#y_predictions = predict(x_predictions,w,b)
#
#
#test_predictions_table = pd.DataFrame({
#    'Economy': x_test.flatten(),
#    'Test Score': y_test.flatten(),
#    'Predicted Score': predict(x_test.flatten(),w,b).flatten(),
#})
#
#print(test_predictions_table.head(10))

