import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

#Importing the dataset
dataset = pd.read_csv(r'Datasets\Bible_Commandments_Clustering.csv')
X = dataset.iloc[:, [1, 3]].values
actual_values_X = dataset.iloc[:, [1, 3]].values #This is for the actual names
print(X)

#Encoding the two columns
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])
X[:, 1] = le.fit_transform(X[:, 1])

print(X)
print(X.shape)

#Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #WCSS = Sum squared distance between each point and the centroid in a cluster
plt.show()


#Training the K-Means model on the dataset
kmeans = KMeans(n_clusters=6, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)
print(y_kmeans.dtype)

#Visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, c = 'orange', label = 'Cluster 6')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Bible Commandments')
plt.xlabel('Commandments')
plt.ylabel('Scripture Reference')
plt.legend()
#mplcursors.cursor(hover=True) #Uncommment this to hover labels for exact points
plt.show()

#Converting y_kmeans, X, and actual_values_X to pandas dataframes
y_kmeans = pd.DataFrame(y_kmeans, columns=['Which Cluster'])
X = pd.DataFrame(X, columns=['Numerical Commandment Reference', 'Numerical Bible Verse Reference'])
actual_values_X = pd.DataFrame(actual_values_X, columns=['Commandment', 'Bible Verse'])
print(y_kmeans)
print(X)
print(actual_values_X)

#Join all the dataframes and save to a csv file
X = pd.concat([y_kmeans, actual_values_X, X], axis=1)
X.to_csv('Bible_Commandments_Clustering.csv', index=False)