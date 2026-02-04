
# SCENARIO 2 – POLYNOMIAL REGRESSION

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

print("Mithra Ravi - 24BAD070")
# 2. Load and clean Auto MPG dataset
df = pd.read_csv("auto-mpg.csv")

# Replace '?' with NaN and convert horsepower to numeric
df['horsepower'] = df['horsepower'].replace('?', np.nan)
df['horsepower'] = pd.to_numeric(df['horsepower'])

# 4. Handle missing values
df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)


# 3. Select feature and target
X = df[['horsepower']]
y = df['mpg']


# 6. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# Store results
degrees = [2, 3, 4]
results = {}


# 8–10. Train Polynomial Regression models
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)
    
    results[degree] = {
        'model': model,
        'poly': poly,
        'train_error': mean_squared_error(y_train, y_train_pred),
        'test_error': mse,
        'rmse': rmse,
        'r2': r2
    }
    
    print(f"\nPolynomial Degree {degree}")
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R²  :", r2)


# 11. Compare model performance
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df[['train_error', 'test_error', 'rmse', 'r2']])


# 12. Ridge Regression (degree = 4)
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_poly, y_train)

ridge_pred = ridge.predict(X_test_poly)

print("\nRidge Regression (Degree 4)")
print("MSE :", mean_squared_error(y_test, ridge_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, ridge_pred)))
print("R²  :", r2_score(y_test, ridge_pred))


# ================================
# VISUALIZATIONS
# ================================

# Polynomial Curve Fitting
plt.figure()
X_range = np.linspace(X_scaled.min(), X_scaled.max(), 100).reshape(-1, 1)

for degree in degrees:
    poly = results[degree]['poly']
    model = results[degree]['model']
    X_range_poly = poly.transform(X_range)
    y_range_pred = model.predict(X_range_poly)
    plt.plot(X_range, y_range_pred, label=f"Degree {degree}")

plt.scatter(X_scaled, y, alpha=0.3)
plt.xlabel("Horsepower (Scaled)")
plt.ylabel("MPG")
plt.title("Polynomial Regression Curve Fitting")
plt.legend()
plt.show()


# Training vs Testing Error Comparison
train_errors = [results[d]['train_error'] for d in degrees]
test_errors = [results[d]['test_error'] for d in degrees]

plt.figure()
plt.plot(degrees, train_errors, marker='o', label="Training Error")
plt.plot(degrees, test_errors, marker='o', label="Testing Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Training vs Testing Error")
plt.legend()
plt.show()
