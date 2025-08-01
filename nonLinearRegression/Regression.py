import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RegressionModel:
    def __init__(self,data,labels,degree=0,normalize=False):
        self.data = self.prepare_data(data,degree,normalize)
        self.labels = labels
        self.weights = np.zeros(degree+1)

    def normalize(self,data,normalize):
        if not normalize:
            return data
        myu = np.mean(data[:,1:],axis=0)
        sigma = np.std(data[:,1:],axis=0)
        data[:,1:] = (data[:,1:] - myu) / sigma
        return data

    def prepare_data(self,data,degree,normalize):
        x = np.array(data).flatten()
        m = x.shape[0]
        poly = np.ones((m,degree+1))
        for d in range(1,degree+1):
            poly[:,d] = x ** d
        return self.normalize(poly,normalize)

    def predict(self,w):
        return np.dot(self.data,w)

    def cost(self):
        m = self.data.shape[0]
        predictions = self.predict(self.weights)
        errors = predictions - self.labels
        cost = (1/(2*m)) * np.dot(errors,errors)
        return cost

    def find_grad(self):
        m = self.data.shape[0]
        predictions = self.predict(self.weights)
        error = predictions - self.labels
        grad = (1/m)*np.dot(self.data.T,error)
        return grad

    def train(self,alpha,epochs):
        for i in range(epochs):
            dw = self.find_grad()
            self.weights -= alpha*dw

            if i%100 == 0:
                print(f"Epoch {i}, Cost: {self.cost():.4f}")

        return self.weights

