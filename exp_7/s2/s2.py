print("Mithra Ravi - 24BAD070")

# ==============================
# Step 1: Load Dataset
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

df = pd.read_csv("mall_cus.csv")

# ==============================
# Step 2: Preprocessing & Scaling
# ==============================

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# Step 3: Apply GMM
# ==============================

k = 5
gmm = GaussianMixture(n_components=k, random_state=42)
gmm.fit(X_scaled)

print("\nGMM model trained successfully!")
from sklearn.metrics import silhouette_score

# ==============================
# Step 4: Choose number of components using AIC & BIC
# ==============================

aic_values = []
bic_values = []
K_range = range(1, 11)

for k in K_range:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    
    aic_values.append(gmm.aic(X_scaled))
    bic_values.append(gmm.bic(X_scaled))

# Plot AIC & BIC
plt.figure()
plt.plot(K_range, aic_values, marker='o', label='AIC')
plt.plot(K_range, bic_values, marker='o', label='BIC')
plt.xlabel('Number of Components')
plt.ylabel('Score')
plt.title('AIC & BIC for GMM')
plt.legend()
plt.show()

# ==============================
# Choose optimal K (example: pick lowest BIC)
# ==============================
optimal_k = K_range[np.argmin(bic_values)]
print("Optimal K (based on BIC):", optimal_k)

# ==============================
# Step 5: Fit GMM using optimal K
# ==============================

gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm.fit(X_scaled)

# ==============================
# Step 6: Predict cluster probabilities
# ==============================

probabilities = gmm.predict_proba(X_scaled)

print("\nCluster Probabilities (first 5 rows):")
print(probabilities[:5])

# ==============================
# Step 7: Assign clusters (highest probability)
# ==============================

labels = np.argmax(probabilities, axis=1)
df['GMM_Cluster'] = labels

print("\nClustered Data:")
print(df.head())

# ==============================
# Step 8: Visualization
# ==============================

plt.figure()

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)

# Plot means (centers)
means = gmm.means_
plt.scatter(means[:, 0], means[:, 1], s=200, marker='X')

plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.title('GMM Clustering')

plt.show()

# ==============================
# Evaluation Metrics
# ==============================

print("\nLog-Likelihood:", gmm.score(X_scaled))
print("AIC:", gmm.aic(X_scaled))
print("BIC:", gmm.bic(X_scaled))
print("Silhouette Score:", silhouette_score(X_scaled, labels))
