import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict(x,w1,b1):
    return x*w1 + b1

def cost_function(x,w1,b1,out):
    m = len(x)
    y_hat = predict(x,w1,b1)
    cost = (1 / (2 * m)) * np.sum((y_hat - out) ** 2)
    return cost

def find_gradient(x,y,w1,b1):
    m = len(x)
    y_hat = predict(x,w1,b1)
    error = y_hat - y
    dw = (1/m)*np.dot(error,x)
    db = (1/m)*np.sum(error)
    return dw, db

def gradient_descend(x,y,w1,b1,alpha,n_iter):
    w = w1
    b = b1
    for i in range(n_iter):
        dw, db = find_gradient(x,y,w,b)
        w -= alpha*dw
        b -= alpha*db
    return w,b

data: pd.DataFrame = pd.read_csv('2017.csv')
train_data: pd.DataFrame = data.sample(frac=0.8)
test_data: pd.DataFrame = data.drop(train_data.index)

x_train = train_data['Economy..GDP.per.Capita.'].values
y_train = train_data['Happiness.Score'].values

x_test = test_data['Economy..GDP.per.Capita.'].values
y_test = test_data['Happiness.Score'].values

w = 0.0  
b = 0.0 

cost = cost_function(x_train, y_train, w, b)
print('Initial cost: {:.2f}'.format(cost))

w,b = gradient_descend(x_train,y_train,w,b,0.01,50000)

cost = cost_function(x_train, y_train, w, b)
print('Optimized cost: {:.2f}'.format(cost))

x_predictions = np.linspace(x_train.min(), x_train.max(), 100)
y_predictions = predict(x_predictions,w,b)


test_predictions_table = pd.DataFrame({
    'Economy': x_test.flatten(),
    'Test Score': y_test.flatten(),
    'Predicted Score': predict(x_test.flatten(),w,b).flatten(),
})

print(test_predictions_table.head(10))


plt.scatter(x_train, y_train, label='Training Dataset')
plt.scatter(x_test, y_test, label='Test Dataset')
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.xlabel("Economy")
plt.ylabel("Happiness")
plt.title('Countries Happines')
plt.legend()
plt.show()
