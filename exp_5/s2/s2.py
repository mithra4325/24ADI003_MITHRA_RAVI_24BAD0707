# ==============================================
# mithra ravi - 24bad070
# SCENARIO 2 – DECISION TREE CLASSIFIER
# Loan Approval Prediction
# ==============================================

# ================================
# STEP 1: Import Required Libraries
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

print("Mithra Ravi - 24BAD070")

# ================================
# STEP 2: Load Dataset
# ================================

df = pd.read_csv("loan.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values Before Handling:")
print(df.isnull().sum())


# ================================
# STEP 3: Data Preprocessing
# ================================

# Drop Loan_ID (not useful for prediction)
df.drop("Loan_ID", axis=1, inplace=True)

# Fill numerical columns with median
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())

# Fill categorical columns with mode
categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Handling:")
print(df.isnull().sum())


# ================================
# Encode Categorical Variables
# ================================

le = LabelEncoder()

categorical_cols = ['Gender', 'Married', 'Dependents',
                    'Education', 'Self_Employed',
                    'Property_Area', 'Loan_Status']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("\nDataset After Encoding:")
print(df.head())


# ================================
# Select Required Features
# ================================

features = ['ApplicantIncome', 'LoanAmount',
            'Credit_History', 'Education',
            'Property_Area']

X = df[features]
y = df['Loan_Status']

print("\nFeature Matrix Shape:", X.shape)
print("Target Vector Shape:", y.shape)


# ================================
# STEP 4: Train-Test Split
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)


# ================================
# STEP 5: Train Decision Tree
# ================================

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

print("\nDecision Tree Model Trained")


# ================================
# STEP 6: Predictions
# ================================

y_pred = dt.predict(X_test)

print("\nPredicted Loan Status:")
print(y_pred)


# ================================
# STEP 7: Model Evaluation
# ================================

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ================================
# STEP 9: Feature Importance
# ================================

import pandas as pd

feature_importance = pd.Series(
    dt.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot Feature Importance
plt.figure()
feature_importance.plot(kind='bar')
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.show()

# ================================
# STEP 10: Detect Overfitting
# ================================

train_accuracy = accuracy_score(y_train, dt.predict(X_train))
test_accuracy = accuracy_score(y_test, dt.predict(X_test))

print("\nTraining Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# ================================
# STEP 11: Compare Shallow vs Deep Tree
# ================================

# Shallow Tree
dt_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_shallow.fit(X_train, y_train)

shallow_train_acc = accuracy_score(y_train, dt_shallow.predict(X_train))
shallow_test_acc = accuracy_score(y_test, dt_shallow.predict(X_test))

print("\nShallow Tree (max_depth=3)")
print("Training Accuracy:", shallow_train_acc)
print("Testing Accuracy:", shallow_test_acc)

# Deep Tree (already trained as dt)
deep_train_acc = accuracy_score(y_train, dt.predict(X_train))
deep_test_acc = accuracy_score(y_test, dt.predict(X_test))

print("\nDeep Tree (No Depth Limit)")
print("Training Accuracy:", deep_train_acc)
print("Testing Accuracy:", deep_test_acc)

# Regularized Decision Tree
dt_regularized = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

dt_regularized.fit(X_train, y_train)

train_acc_reg = accuracy_score(y_train, dt_regularized.predict(X_train))
test_acc_reg = accuracy_score(y_test, dt_regularized.predict(X_test))

print("\nRegularized Tree")
print("Training Accuracy:", train_acc_reg)
print("Testing Accuracy:", test_acc_reg)

# ================================
# CONFUSION MATRIX
# ================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predictions using regularized model
y_pred_reg = dt_regularized.predict(X_test)

cm = confusion_matrix(y_test, y_pred_reg)

print("\nConfusion Matrix:")
print(cm)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Regularized Tree")
plt.show()

# ================================
# TREE STRUCTURE PLOT
# ================================

from sklearn.tree import plot_tree

plt.figure(figsize=(15,10))

plot_tree(
    dt_regularized,
    feature_names=X.columns,
    class_names=["Rejected", "Approved"],
    filled=True
)

plt.title("Decision Tree Structure (Regularized)")
plt.show()
