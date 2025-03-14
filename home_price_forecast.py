import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("train.csv")

# Select relevant features
df_selected = df[['GrLivArea', 'YearBuilt', 'OverallQual', 'TotalBsmtSF', 'GarageCars', 'SalePrice']]

# Visualize relationships
sns.pairplot(df_selected)
plt.show()

# Define features and target
X = df_selected[['GrLivArea', 'YearBuilt', 'OverallQual', 'TotalBsmtSF', 'GarageCars']]
y = df_selected['SalePrice']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
lr_model = LinearRegression()
svm_model = SVR()
dt_model = DecisionTreeRegressor(random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train models
lr_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Compute MSE
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_svm = mean_squared_error(y_test, y_pred_svm)
mse_dt = mean_squared_error(y_test, y_pred_dt)
mse_rf = mean_squared_error(y_test, y_pred_rf)

# Store results in a table
mse_results = pd.DataFrame({
    "Model": ["Linear Regression", "SVM", "Decision Tree", "Random Forest"],
    "MSE": [mse_lr, mse_svm, mse_dt, mse_rf]
})

print(mse_results)

# Visualize actual vs predicted values for Linear Regression
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.6)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Linear Regression: Actual vs. Predicted Sale Prices")
plt.axline([0, 0], [1, 1], color='red', linestyle='dashed')  # Reference line
plt.show()
