import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('iris.csv')

x_axis = "petal_length"
y_axis = "petal_width"
iris_types = ['SETOSA', 'NOT_SETOSA']

num_examples = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
y_train = (data['class'] == "SETOSA").astype(int).values.reshape((num_examples, 1))
w = np.zeros((2,1))
b = 0.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(x, w, b):
    z = np.dot(x,w) + b
    return sigmoid(z)

def predict(x, w, b, th=0.5):
    probs = predict_proba(x, w, b)
    return (probs >= th).astype(int)

def find_loss(y_h,y,eps=1e-15):
    y_h = np.clip(y_h,eps,1-eps)
    return -1*(y*np.log(y_h) + (1 - y)*np.log(1 - y_h))

def find_cost(y_h,y):
    m = y_h.shape[0]
    cost = (1/m) * np.sum(find_loss(y_h,y))
    return cost

def compute_gradients(x, y, y_hat):
    m = y.shape[0]
    error = y_hat - y
    dw = (1/m)*np.dot(x.T,error)
    db = (1/m)*np.sum(error)
    return dw,db

def gradient_descent(x,y,w1,b1,alpha,epochs):
    w=w1
    b=b1
    for i in range(epochs):
        y_hat = predict_proba(x, w, b)
        dw, db = compute_gradients(x, y, y_hat)
        w -= alpha * dw
        b -= alpha * db
    return w, b

y_hat = predict_proba(x_train,w,b)
cost = find_cost(y_hat,y_train)
print("{:.4f}".format(cost))

w,b = gradient_descent(x_train,y_train,w,b,0.01,10000)
y_hat = predict_proba(x_train,w,b)
cost = find_cost(y_hat,y_train)
print("{:.4f}".format(cost))

y_pred = predict(x_train, w, b)
acc = np.mean(y_pred == y_train)
print("Training Accuracy: {:.2f}%".format(acc*100))

plt.scatter(
    data[x_axis][data['class'] == iris_types[0]],
    data[y_axis][data['class'] == iris_types[0]],
    label=iris_types[0]
)
plt.scatter(
    data[x_axis][data['class'] != iris_types[0]],
    data[y_axis][data['class'] != iris_types[0]],
    label=iris_types[1]
)

x_vals = np.linspace(x_train[:, 0].min(),x_train[:, 0].max(), 100)
y_vals = -(w[0] * x_vals + b) / w[1]

plt.plot(x_vals, y_vals, color='black',lw=3, label='Boundary')

plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend()
plt.show()

