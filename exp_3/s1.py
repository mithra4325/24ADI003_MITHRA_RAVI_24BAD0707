# SCENARIO 1 – MULTILINEAR REGRESSION

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

print("Mithra Ravi - 24BAD070")
# 2. Load the dataset
df = pd.read_csv("StudentsPerformance.csv")

print("Dataset Columns:")
print(df.columns)


# 3. Encode categorical variables
df_encoded = pd.get_dummies(
    df,
    columns=[
        'gender',
        'race/ethnicity',
        'parental level of education',
        'lunch',
        'test preparation course'
    ],
    drop_first=True
)


# 4. Compute Target Variable (Final Exam Score)
df_encoded['final_exam_score'] = (
    df_encoded['math score'] +
    df_encoded['reading score'] +
    df_encoded['writing score']
) / 3


# 5. Handle missing values
df_encoded.fillna(df_encoded.mean(), inplace=True)


# 6. Select input features and target
X = df_encoded.drop(
    ['math score', 'reading score', 'writing score', 'final_exam_score'],
    axis=1
)

y = df_encoded['final_exam_score']


# 7. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# 9. Train Multilinear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)


# 10. Prediction
y_pred = model.predict(X_test)


# 11. Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MSE :", mse)
print("RMSE:", rmse)
print("R² Score:", r2)


# 12. Regression Coefficients Analysis
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nRegression Coefficients:")
print(coefficients)


# 13. Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

print("\nRidge Regression R²:", r2_score(y_test, ridge_pred))


# 14. Lasso Regression
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

print("Lasso Regression R²:", r2_score(y_test, lasso_pred))


# ================================
# VISUALIZATIONS
# ================================

# Predicted vs Actual
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Final Exam Score")
plt.ylabel("Predicted Final Exam Score")
plt.title("Predicted vs Actual Exam Scores")
plt.show()


# Coefficient Magnitude Comparison
plt.figure(figsize=(12, 6))
coefficients.set_index('Feature')['Coefficient'].plot(kind='bar')
plt.title("Regression Coefficients")
plt.ylabel("Coefficient Value")
plt.show()


# Residual Distribution
residuals = y_test - y_pred
plt.figure()
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.show()
