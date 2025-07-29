import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('iris.csv')
num_examples = data.shape[0]

def initialize_centroids(x, k):
    indices = np.random.choice(x.shape[0], size=k)
    return x[indices]

def assign_clusters(x, centroids):
    diff = x[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(x, labels, k):
    new_centroids = np.zeros((k, x.shape[1]))
    for i in range(k):
        points = x[labels == i]
        if len(points) > 0:
            new_centroids[i] = points.mean(axis=0)
        else:
            new_centroids[i] = x[np.random.choice(x.shape[0])]
    return new_centroids

def kmeans(x, k, max_iters=100):
    centroids = initialize_centroids(x, k)
    
    for _ in range(max_iters):
        labels = assign_clusters(x, centroids)
        new_centroids = update_centroids(x, labels, k)
        centroids = new_centroids
    
    labels = assign_clusters(x, centroids)
    return centroids, labels


x_axis = 'petal_length'
y_axis = 'petal_width'

x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))


centroids, labels = kmeans(x_train, k=3)

k = centroids.shape[0]
for cluster_id in range(k):
        cluster_points = data[labels == cluster_id]
        plt.scatter(
            cluster_points[x_axis],
            cluster_points[y_axis],
            label='Cluster #' + str(cluster_id)
        )

for centroid_id, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1], c='black', marker='x')


plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend()
plt.show()
