# ==============================================
# mithra ravi - 24bad070
# SCENARIO 1 – K-NEAREST NEIGHBORS (KNN)
# Breast Cancer Classification
# ==============================================

# ================================
# STEP 1: Import Required Libraries
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print ("Mithra Ravi - 24BAD070")

# ================================
# STEP 2: Load Dataset
# ================================

df = pd.read_csv("breast-cancer.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ================================
# STEP 3: Data Inspection & Preprocessing
# ================================

print("\nColumn Names:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())


# ================================
# Select Required Features
# ================================

features = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'smoothness_mean'
]

X = df[features]
y = df['diagnosis']

print("\nFeature Matrix Shape:", X.shape)
print("Target Vector Shape:", y.shape)


# ================================
# STEP 4: Encode Target Labels
# ================================

y = y.map({'M': 1, 'B': 0})

print("\nEncoded Target Values:")
print(y.value_counts())


# ================================
# STEP 5: Feature Scaling
# ================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Applied Successfully")


# ================================
# STEP 6: Train-Test Split
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)


# ================================
# STEP 7: Train Initial KNN Model
# ================================

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("\nInitial KNN Model Trained (K=5)")


# ================================
# STEP 8: Experiment with Different K Values
# ================================

k_values = range(1, 21)
accuracy_scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred_k = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_k)
    accuracy_scores.append(acc)

print("\nAccuracy for Different K Values:")
for k, acc in zip(k_values, accuracy_scores):
    print(f"K = {k} → Accuracy = {acc:.4f}")

best_k = k_values[accuracy_scores.index(max(accuracy_scores))]
print("\nBest K Value:", best_k)


# ================================
# STEP 9: Final Prediction Using Best K
# ================================

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

y_pred = knn_best.predict(X_test)

print("\nPredicted Labels:")
print(y_pred)


# ================================
# STEP 10: Model Evaluation
# ================================

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ================================
# STEP 11: Identify Misclassified Cases
# ================================

misclassified = np.where(y_test != y_pred)

print("\nNumber of Misclassified Cases:", len(misclassified[0]))

print("\nIndices of Misclassified Cases:")
print(misclassified)

print("\nActual vs Predicted (Misclassified Cases):")
for i in misclassified[0]:
    print(f"Actual: {y_test.iloc[i]}, Predicted: {y_pred[i]}")


# ================================
# Confusion Matrix Visualization
# ================================

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Benign', 'Malignant'],
    yticklabels=['Benign', 'Malignant']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ================================
# Accuracy vs K Plot
# ================================

plt.figure()
plt.plot(k_values, accuracy_scores, marker='o')

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.show()


# ================================
# Decision Boundary (Using 2 Features)
# ================================

X_2 = df[['radius_mean', 'texture_mean']]
y_2 = df['diagnosis'].map({'M': 1, 'B': 0})

scaler_2 = StandardScaler()
X_2_scaled = scaler_2.fit_transform(X_2)

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_2_scaled, y_2, test_size=0.2, random_state=42
)

knn_2 = KNeighborsClassifier(n_neighbors=best_k)
knn_2.fit(X_train_2, y_train_2)

x_min, x_max = X_2_scaled[:, 0].min() - 1, X_2_scaled[:, 0].max() + 1
y_min, y_max = X_2_scaled[:, 1].min() - 1, X_2_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)

Z = knn_2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_2_scaled[:, 0], X_2_scaled[:, 1], c=y_2)

plt.xlabel("Radius (Scaled)")
plt.ylabel("Texture (Scaled)")
plt.title(f"KNN Decision Boundary (K={best_k})")
plt.show()
