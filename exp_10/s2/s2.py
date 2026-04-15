print("Mithra Ravi - 24BAD070")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

# LOAD DATA
ratings = pd.read_csv('ratings.csv')[['userId', 'movieId', 'rating']]

# USER-ITEM MATRIX
matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

# NMF MODEL
k = 20
nmf = NMF(n_components=k, init='random', random_state=42, max_iter=200)

W = nmf.fit_transform(matrix)
H = nmf.components_

# RECONSTRUCTION
recon = np.dot(W, H)
recon = pd.DataFrame(recon, index=matrix.index, columns=matrix.columns)

# FIX: CLIP RATINGS
recon = recon.clip(0.5, 5)

# EVALUATION
mask = matrix.values != 0
rmse = np.sqrt(mean_squared_error(matrix.values[mask], recon.values[mask]))
print(f"RMSE: {rmse:.4f}")

# PRECISION & RECALL
def precision_recall_at_k(user_id, k=10, threshold=3.5):
    actual = matrix.loc[user_id]
    predicted = recon.loc[user_id]

    relevant = actual[actual >= threshold].index
    recommended = predicted[actual == 0].nlargest(k).index

    hits = len(set(recommended) & set(relevant))

    precision = hits / k
    recall = hits / len(relevant) if len(relevant) > 0 else 0

    return precision, recall

p, r = precision_recall_at_k(1)
print(f"Precision@10: {p:.4f}")
print(f"Recall@10: {r:.4f}")

# RECOMMENDATIONS
def recommend(user_id, n=10):
    user_ratings = recon.loc[user_id]
    unseen = user_ratings[matrix.loc[user_id] == 0]
    return unseen.nlargest(n)

print("\nTop 10 for User 1:\n", recommend(1))

# ======================
# SEPARATE GRAPHS
# ======================

# 1. Actual vs Predicted
plt.figure()
plt.scatter(matrix.values[mask], recon.values[mask], alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (NMF)")
plt.show()

# 2. Latent Features
plt.figure()
plt.imshow(W[:20, :], aspect='auto')
plt.colorbar()
plt.title("User Latent Features")
plt.xlabel("Features")
plt.ylabel("Users")
plt.show()

# 3. Recommendations
top10 = recommend(1)
plt.figure()
top10.plot(kind='barh')
plt.title("Top 10 Recommendations (User 1)")
plt.xlabel("Predicted Rating")
plt.show()
