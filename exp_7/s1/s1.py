import os
os.environ["OMP_NUM_THREADS"] = "1"

print("Mithra Ravi - 24BAD070")
# ==============================
# Task 1: Import Libraries
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ==============================
# Task 2: Load Dataset
# ==============================
df = pd.read_csv("mall_cus.csv")


# ==============================
# Task 3: Data Preprocessing
# ==============================

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())


# ==============================
# Task 4: Feature Selection
# ==============================

# Selecting relevant features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]


# ==============================
# Feature Scaling
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==============================
# Task 5: Elbow Method
# ==============================

inertia_values = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure()
plt.plot(K_range, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method')
plt.show()

# ==============================
# Task 6: Apply K-Means
# ==============================

k = 5  

kmeans = KMeans(n_clusters=k, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)


# ==============================
# Task 7: Assign Cluster Labels
# ==============================

df['Cluster'] = y_kmeans

print("\nClustered Data:")
print(df.head())


# ==============================
# Task 8: Visualization
# ==============================

plt.figure()

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans)

# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='X')

plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.title('Customer Segments (K-Means Clustering)')

plt.show()


# ==============================
# Task 9: Interpretation
# ==============================

print("\nCluster Means:")
print(df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())


# ==============================
# Evaluation Metrics
# ==============================

from sklearn.metrics import silhouette_score

print("\nInertia:", kmeans.inertia_)
print("Silhouette Score:", silhouette_score(X_scaled, y_kmeans))
