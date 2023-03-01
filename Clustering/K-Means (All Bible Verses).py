import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

#Importing the dataset and preprocesses
dataset = pd.read_csv(r'Datasets\Pre Model\t_kjv.csv')
dataset['com'] = dataset['b'].astype(str) + '' + dataset['c'].astype(str) + '' + dataset['v'].astype(str)
dataset['com'] = dataset['com'].astype(int)
X = dataset.iloc[:, 4:].values
passages = dataset.iloc[:, [4, 5]].values

#Encodes the words into numbers
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])
X[:, 1] = le.fit_transform(X[:, 1])
print(X)

#Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 15), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #WCSS = Sum squared distance between each point and the centroid in a cluster
plt.show()

#Training the K-Means model on the dataset
kmeans = KMeans(n_clusters=9, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)

#Visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'red', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, c = 'orange', label = 'Cluster 6')
plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s = 100, c = 'gray', label = 'Cluster 7')
plt.scatter(X[y_kmeans == 7, 0], X[y_kmeans == 7, 1], s = 100, c = 'black', label = 'Cluster 8')
plt.scatter(X[y_kmeans == 8, 0], X[y_kmeans == 8, 1], s = 100, c = 'deeppink', label = 'Cluster 9')
#plt.scatter(X[y_kmeans == 9, 0], X[y_kmeans == 9, 1], s = 100, c = 'peru', label = 'Cluster 10')
#plt.scatter(X[y_kmeans == 10, 0], X[y_kmeans == 10, 1], s = 100, c = 'mediumblue', label = 'Cluster 11')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('The Bible')
plt.xlabel('Bible Verse')
plt.ylabel('Scripture')
plt.legend()
plt.show()

y_kmeans = pd.DataFrame(y_kmeans, columns=['Which Cluster'])
X = pd.DataFrame(X, columns=['Numerical Scripture Version', 'Combined Bible Verse Encoded'])
passages = pd.DataFrame(passages, columns=['Passages', 'Combined Bible Verse (B C V)'])
print(y_kmeans)
print(X)
print(passages)

#Join all the dataframes and save to a csv file
X = pd.concat([y_kmeans, passages, X], axis=1)
X.to_csv(r'Datasets\Post Model\Bible_Verses_Clustering.csv', index=False)