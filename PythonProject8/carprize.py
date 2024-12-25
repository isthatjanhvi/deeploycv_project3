import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm

# Sample Data
data = {
    'Engine Size': [2.0, 3.5, 1.8, 2.5, 1.6],
    'Mileage': [100000, 50000, 150000, 75000, 200000],
    'Age of Car': [5, 2, 8, 3, 10],
    'Previous Owners': [1, 2, 1, 1, 3],
    'Price': [20000, 35000, 12000, 28000, 8000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Feature and Target variable
X = df[['Engine Size', 'Mileage', 'Age of Car', 'Previous Owners']]
y = df['Price']

# Add constant for intercept in statsmodels
X = sm.add_constant(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Linear Regression model
model = sm.OLS(y_train, X_train).fit()

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print results
print(f"R-squared: {r2}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# p-Statistic for each coefficient
print(f"p-Statistics: {model.pvalues}")
