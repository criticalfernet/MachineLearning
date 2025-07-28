import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('iris.csv')

x_axis = "petal_length"
y_axis = "petal_width"
iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']
classifiers = []

num_examples = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)

x_train_n = (x_train-mean)/std



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

def multi_predict(x,classifiers):
    m = x.shape[0]
    count_classes = len(classifiers)
    probs = np.zeros((m,count_classes))

    for i,(cls,w,b) in enumerate(classifiers):
        probs[:,i] = predict_proba(x,w,b).flatten()

    class_indices = np.argmax(probs, axis=1)
    class_labels = [classifiers[i][0] for i in class_indices]
    return np.array(class_labels)




for type in iris_types:
    y_train = (data['class'] == type).astype(int).values.reshape(
        (num_examples,1)
    )
    w = np.zeros((2,1))
    b = 0.0
    w,b = gradient_descent(x_train_n,y_train,w,b,0.01,10000)

    classifiers.append((type, w, b))


true_labels = data['class'].values
predicted_labels = multi_predict(x_train_n, classifiers)
accuracy = np.mean(predicted_labels == true_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")


for type in iris_types:
    plt.scatter(
        data[x_axis][data['class'] == type],
        data[y_axis][data['class'] == type],
        label=type
    )

x_vals_n = np.linspace(x_train_n[:, 0].min(),x_train_n[:, 0].max(), 100)
x_vals = x_vals_n*std[0] + mean[0]

for cls_name, w, b in classifiers:
    y_vals_n = -(w[0] * x_vals_n + b) / w[1]
    y_vals = y_vals_n*std[1] + mean[1]
    plt.plot(x_vals, y_vals,linestyle='--', label=cls_name)

plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend()
plt.show()

