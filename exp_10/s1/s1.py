print("Mithra Ravi - 24BAD070")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error

# STEP 1: LOAD DATA
ratings_df = pd.read_csv('ratings.csv')
ratings_df = ratings_df[['userId', 'movieId', 'rating']]
print(f"Dataset loaded: {ratings_df.shape}")

# STEP 2: USER-ITEM MATRIX
user_item_matrix = ratings_df.pivot_table(
    index='userId',
    columns='movieId',
    values='rating',
    fill_value=0
)
print(f"Matrix: {user_item_matrix.shape}")

# STEP 3: NORMALIZATION
user_means = (user_item_matrix.sum(axis=1) / (user_item_matrix != 0).sum(axis=1)).values
user_means = np.nan_to_num(user_means)

normalized_matrix = user_item_matrix.copy()
for i in range(normalized_matrix.shape[0]):
    normalized_matrix.iloc[i, :] = normalized_matrix.iloc[i, :].apply(
        lambda x: x - user_means[i] if x != 0 else 0
    )

# STEP 4: TEST K VALUES
k_values = [5, 10, 15, 20, 30, 50, 60, 70, 80]
rmse_scores, mae_scores = [], []

print("\nK vs Error:")
for k in k_values:
    svd = TruncatedSVD(n_components=k, random_state=42)
    U = svd.fit_transform(normalized_matrix)
    V = svd.components_
    
    reconstructed = np.dot(U, V)
    reconstructed_df = pd.DataFrame(reconstructed, index=user_item_matrix.index, columns=user_item_matrix.columns)
    
    for i in range(reconstructed_df.shape[0]):
        reconstructed_df.iloc[i, :] += user_means[i]
    reconstructed_df = reconstructed_df.clip(0.5, 5.0)
    
    mask = (user_item_matrix != 0).values
    rmse = np.sqrt(mean_squared_error(user_item_matrix.values[mask], reconstructed_df.values[mask]))
    mae = mean_absolute_error(user_item_matrix.values[mask], reconstructed_df.values[mask])
    
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    print(f"k={k}: RMSE={rmse:.4f}, MAE={mae:.4f}")

# STEP 5: FINAL MODEL
k = 50
svd = TruncatedSVD(n_components=k, random_state=42)
U = svd.fit_transform(normalized_matrix)
V = svd.components_

reconstructed = np.dot(U, V)
reconstructed_df = pd.DataFrame(reconstructed, index=user_item_matrix.index, columns=user_item_matrix.columns)

for i in range(reconstructed_df.shape[0]):
    reconstructed_df.iloc[i, :] += user_means[i]
reconstructed_df = reconstructed_df.clip(0.5, 5.0)

# STEP 6: EVALUATION
mask = (user_item_matrix != 0).values
actual = user_item_matrix.values[mask]
predicted = reconstructed_df.values[mask]

rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)

print(f"\nFinal -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# STEP 7: RECOMMENDATIONS
def recommend(user_id):
    preds = reconstructed_df.loc[user_id]
    unrated = preds[user_item_matrix.loc[user_id] == 0]
    return unrated.nlargest(10)

print("\nTop Recommendations:")
for user in [1, 2]:
    print(f"\nUser {user}:")
    recs = recommend(user)
    for i, (movie, rating) in enumerate(recs.items(), 1):
        print(f"{i}. Movie {int(movie)} -> {rating:.2f}")

# STEP 8: VISUALS (UNCHANGED LOGIC, CLEANED LABELS)
# ===== VISUAL 1: HEATMAP (ORIGINAL vs RECONSTRUCTED) =====
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(user_item_matrix.iloc[:15, :15], cmap='YlOrRd')
plt.title('Original Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(reconstructed_df.iloc[:15, :15], cmap='YlOrRd')
plt.title('Reconstructed Matrix (k=20)')

plt.tight_layout()
plt.show()


# ===== VISUAL 2: ERROR vs LATENT FACTORS =====
plt.figure(figsize=(6, 4))

plt.plot(k_values, rmse_scores, marker='o', label='RMSE')
plt.plot(k_values, mae_scores, marker='s', label='MAE')

plt.xlabel('k (Latent Factors)')
plt.ylabel('Error')
plt.title('Error vs Latent Factors')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ===== VISUAL 3: TOP RECOMMENDED MOVIES =====
plt.figure(figsize=(6, 4))

recommend(1).plot(kind='barh')
plt.title('Top 10 Recommendations (User 1)')
plt.xlabel('Predicted Rating')

plt.tight_layout()
plt.show()
