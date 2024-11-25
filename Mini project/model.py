import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Boston Housing dataset from sklearn
boston = load_boston()

# Convert the dataset into a pandas DataFrame
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV'] = boston.target  # Target variable

# Split the data into features (X) and target (y)
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
