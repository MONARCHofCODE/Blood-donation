import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic sales data for illustration
# For real-world applications, load your actual sales data from a CSV or database

# Let's assume 'Month' is the independent variable (feature) and 'Sales' is the dependent variable (target)
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Sales': [230, 245, 258, 265, 280, 290, 300, 320, 330, 340, 350, 360]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Feature and target variables
X = df[['Month']]  # Independent variable
y = df['Sales']    # Dependent variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Predict the sales using the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Plot the actual vs predicted sales
plt.scatter(X_test, y_test, color='blue', label='Actual Sales')
plt.plot(X_test, y_pred, color='red', label='Predicted Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Sales Forecasting - Actual vs Predicted')
plt.legend()
plt.show()

# Make a prediction for the next month
next_month = np.array([[13]])  # Predict sales for the 13th month
predicted_sales = model.predict(next_month)
print(f'Predicted sales for next month (Month 13): {predicted_sales[0]}')

