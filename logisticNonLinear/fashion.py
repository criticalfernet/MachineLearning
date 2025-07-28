import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

data = pd.read_csv('fashion.csv').head(1000)
#data = data.iloc[:, [0] + list(np.random.choice(range(1, 785), 500, replace=False))]

x_train = data.drop(columns=["label"]).values
y_train = data["label"].values.reshape((-1, 1))
x_train = x_train / 255.0


#validities = [0, 1]
DEGREE = 3
#
def transform_polynomial_features(X, degree):
    X = np.array(X)
    m, n = X.shape
    X_poly = []

    for i in range(m):
        row = [1.0]  # bias term
        stack = [(X[i, j], 1, j) for j in range(n)]

        while stack:
            val, deg, idx = stack.pop()
            if deg <= degree:
                row.append(val)
                for j in range(idx, n):
                    new_val = val * X[i, j]
                    stack.append((new_val, deg + 1, j))

        X_poly.append(row)

    # pad shorter rows with zeros to match longest row
    max_len = max(len(row) for row in X_poly)
    for row in X_poly:
        while len(row) < max_len:
            row.append(0.0)

    return np.array(X_poly)
#
#
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w, b):
    return sigmoid(np.dot(X, w) + b)

def compute_loss(y_hat, y, eps=1e-15):
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -1 * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def compute_cost(y_hat, y):
    m = y.shape[0]
    return (1/m) * np.sum(compute_loss(y_hat, y))

def compute_gradients(X, y, y_hat):
    m = y.shape[0]
    error = y_hat - y
    dw = (1/m) * np.dot(X.T, error)
    db = (1/m) * np.sum(error)
    return dw, db

def gradient_descent(X, y, w, b, alpha, epochs):
    for _ in range(epochs):
        y_hat = predict_proba(X, w, b)
        dw, db = compute_gradients(X, y, y_hat)
        w -= alpha * dw
        b -= alpha * db
    return w, b



x_train_poly = transform_polynomial_features(x_train, degree=1)

classifiers = []
num_classes = 10
m, n = x_train_poly.shape

for digit in range(num_classes):
    y_binary = (y_train == digit).astype(int)

    w = np.zeros((n, 1))
    b = 0.0
    w, b = gradient_descent(x_train_poly, y_binary, w, b, alpha=0.05, epochs=1000)

    classifiers.append((digit, w, b))


def multi_class_predict(X, classifiers):
    m = X.shape[0]
    k = len(classifiers)
    probs = np.zeros((m, k))

    for i, (digit, w, b) in enumerate(classifiers):
        probs[:, i] = predict_proba(X, w, b).flatten()

    predictions = np.argmax(probs, axis=1)
    return predictions

y_pred = multi_class_predict(x_train_poly, classifiers)
accuracy = np.mean(y_pred == y_train.flatten())
print(f"Accuracy: {accuracy * 100:.2f}%")

#
#
#for validity in validities:
#    plt.scatter(
#        data[x_axis][data['validity'] == validity],
#        data[y_axis][data['validity'] == validity],
#        label=validity
#    )
#
#x_vals = np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 200)
#y_vals = np.linspace(x_train[:, 1].min(), x_train[:, 1].max(), 200)
#x_grid, y_grid = np.meshgrid(x_vals, y_vals)
#
#grid_points = np.array([x_grid.ravel(), y_grid.ravel()]).T
#grid_poly = transform_polynomial_features(grid_points, DEGREE)
#
#z = predict_proba(grid_poly, w, b).reshape(x_grid.shape)
#
#plt.contour(x_grid, y_grid, z, levels=[0.5], linewidths=2, colors='black')
#
#plt.xlabel(x_axis)
#plt.ylabel(y_axis)
#plt.legend()
#plt.show()

