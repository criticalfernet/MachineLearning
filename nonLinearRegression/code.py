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
#
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

data: pd.DataFrame = pd.read_csv('non-linear.csv')

x = data[['x']].values
y = data['y'].values

w = [0.0]
b = 0.0

w,b,cost_h = gradient_descend(x,y,w,b,0.001,50)

#plt.plot(range(50), cost_h)
#plt.xlabel('Iterations')
#plt.ylabel('Cost')
#plt.title('Gradient Descent Progress')
#plt.show()

print(w,b)

plt.plot(x,y)
x_ran = np.linspace(x.min(),x.max(),100)
plt.plot(x_ran,np.dot(x_ran,w[0])+b,'r')
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

