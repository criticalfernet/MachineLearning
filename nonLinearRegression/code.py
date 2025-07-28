import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize_features(X):
    x_norm = X
    myu = np.mean(X[:, 1:], axis=0)
    sigma = np.std(X[:, 1:], axis=0)
    x_norm[:, 1:] = (X[:, 1:] - myu) / sigma
    return x_norm


def transform_polynomial_features(X,degree):
    X = np.array(X).flatten()
    m = X.shape[0]
    X_poly = np.ones((m, degree + 1))
    for d in range(1, degree + 1):
        X_poly[:, d] = X ** d
    return X_poly


def find_cost(X,y,theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.dot(errors, errors)
    return cost


def gradient_descent(X, y, theta, alpha, num):
    m = len(y)
    for i in range(num):
        predictions = X.dot(theta)
        errors = predictions - y
        gradients = (1 / m) * X.T.dot(errors)
        theta -= alpha * gradients
    return theta


data: pd.DataFrame = pd.read_csv('non-linear.csv')

x = data[['x']].values
y = data['y'].values
degree = 5
x_poly = transform_polynomial_features(x,degree)
x_norm = normalize_features(x_poly)

theta = np.zeros(degree + 1)

cost = find_cost(x_norm,y,theta)
print("{:.2f}".format(cost))

theta = gradient_descent(x_norm,y,theta,0.01,1000)

cost = find_cost(x_norm,y,theta)
print("{:.2f}".format(cost))


x_range = np.linspace(min(x), max(x), 100000)
x_r_poly = transform_polynomial_features(x_range,degree)
x_r_norm = normalize_features(x_r_poly)
plt.plot(x, y)
plt.plot(x_range, x_r_norm.dot(theta), color='red')
plt.show()

