import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets.samples_generator import make_blobs

X, y_true = make_blobs(n_samples = 200, centers = 5, cluster_std=0.75, random_state = 0)

plt.scatter(X[:, 0],X[:, 1], s = 60)

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 5)
km.fit(X)

y_km = km.predict(X)
plt.scatter(X[:, 0],X[:, 1], c = y_km, s = 50, cmap = 'viridis')

centers = km.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s = 200, alpha = 0.5)
plt.savefig('Print.png')