import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, k=2):
        self.k = k

    def initializ_centroids(self, X):
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.k]]
        return centroids

    def compute_centroids(self, X, classifications):
        centroids = np.zeros((self.k, X.shape[1]))
        for k in range(self.k):
            centroids[k, :] = np.mean(X[classifications == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.k))
        for k in range(self.k):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)

        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def fit(self, dataSet):
        self.dataSet = dataSet
        self.centroids = self.initializ_centroids(dataSet)
        changed = True
        while changed:
            old_centroids = self.centroids
            distance = self.compute_distance(dataSet, old_centroids)
            self.classifications = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(dataSet, self.classifications)

            if np.all(old_centroids == self.centroids):
                changed = False

        return self.centroids, self.classifications

class PlotClusterings:
    def __init__(self, data):
        self.data = data

    def colors(self, index):
        colors_hex_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        return colors_hex_list[index]

    def plot(self):
        m,n = self.data.shape

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle('Clustering by K-Means algorithm')

        k=2
        model = Kmeans(k)
        centroids, classifications = model.fit(self.data)
        indexs = []
        indexs2 = []

        for i in range(m):
            markIndex = int(classifications[i])
            if markIndex==8:
                indexs.append(i)

            if markIndex==3:
                indexs2.append(i)

            ax1.scatter(self.data[i,0],self.data[i,1], marker="+", color=self.colors(markIndex))

        for i in range(k):
            ax1.scatter(centroids[i,0],centroids[i,1], marker="<", color='black', s=40, linewidths=5,
            label='centroid')
        ax1.set_title('K = {}'.format(k))

        k = 4
        model = Kmeans(k)
        centroids, classifications = model.fit(self.data)

        for i in range(m):
            markIndex = int(classifications[i])
            ax2.scatter(self.data[i,0],self.data[i,1], marker="+", color=self.colors(markIndex))

        for i in range(k):
            ax2.scatter(centroids[i,0],centroids[i,1], marker="<", color='black', s=40, linewidths=5,
            label='centroid')
        ax2.set_title('K = {}'.format(k))

        k = 7
        model = Kmeans(k)
        centroids, classifications = model.fit(self.data)

        for i in range(m):
            markIndex = int(classifications[i])
            ax3.scatter(self.data[i,0],self.data[i,1], marker="+", color=self.colors(markIndex))

        for i in range(k):
            ax3.scatter(centroids[i,0],centroids[i,1], marker="<", color='black', s=40, linewidths=5,
            label='centroid')
        ax3.set_title('K = {}'.format(k))

        k = 10
        model = Kmeans(k)
        centroids, classifications = model.fit(self.data)
        for i in range(m):
            markIndex = int(classifications[i])
            ax4.scatter(self.data[i,0],self.data[i,1], marker="+", color=self.colors(markIndex))

        for i in range(k):
            ax4.scatter(centroids[i,0],centroids[i,1], marker="<", color='black', s=40, linewidths=5,
            label='centroid')
        ax4.set_title('K = {}'.format(k))

        plt.show() 

class DrawFacesImages:
    def draw(self, data, index, columns=8, rows=8):
        fig=plt.figure(figsize=(10, 10))
        for i in range(1, columns*rows +1):
            image_array = data[np.random.choice(index)]
            img = image_array.reshape(28,20)

            fig.add_subplot(rows, columns, i)
            plt.axis('off')
            plt.imshow(img)
        plt.show()


def load_dataset(inputData, delim, data_type='float64'):
    data = np.genfromtxt(inputData, delimiter=delim, dtype = data_type)
    return data

def main():
    faces = load_dataset('frey-faces.csv',' ')
    model = PlotClusterings(faces)
    model.plot()

if __name__ == '__main__':
  main()