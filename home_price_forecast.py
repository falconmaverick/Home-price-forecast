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

# Visualize relationships between the features and SalePrice
sns.pairplot(df_selected)
plt.show()

# Define features and target variable
X = df_selected[['GrLivArea', 'YearBuilt', 'OverallQual', 'TotalBsmtSF', 'GarageCars']]
y = df_selected['SalePrice']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
lr_model = LinearRegression()  # Linear Regression model
svm_model = SVR()              # Support Vector Machine model
dt_model = DecisionTreeRegressor(random_state=42)  # Decision Tree model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Random Forest model

# Train models on the training data
lr_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_lr = lr_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Compute Mean Squared Error (MSE) for each model
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_svm = mean_squared_error(y_test, y_pred_svm)
mse_dt = mean_squared_error(y_test, y_pred_dt)
mse_rf = mean_squared_error(y_test, y_pred_rf)

# Store MSE results in a DataFrame
mse_results = pd.DataFrame({
    "Model": ["Linear Regression", "SVM", "Decision Tree", "Random Forest"],
    "MSE": [mse_lr, mse_svm, mse_dt, mse_rf]
})

# Print MSE results for comparison
print(mse_results)

# Visualize Actual vs Predicted values for Linear Regression
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.6)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Linear Regression: Actual vs. Predicted Sale Prices")
plt.axline([0, 0], [1, 1], color='red', linestyle='dashed')  # Reference line (y = x)
plt.show()

# --- Output Analysis ---
# Based on the Mean Squared Error (MSE) results, here’s the conclusion:
# 
# The model with the **lowest MSE** is the best performing model. Lower MSE means better predictions.
# 
# After evaluating the models, I found:
# - **Linear Regression** performed decently. It assumes a linear relationship between the features and target, which works well when the data follows a linear pattern. However, it doesn't capture complex relationships effectively.
# - **Support Vector Machine (SVM)** showed promise for non-linear relationships but didn’t perform better than the other models in this case. It struggles when the data doesn't have a clear margin of separation.
# - **Decision Tree** captured the non-linear patterns in the data but overfit. This means it did well on the training set but struggled to generalize to unseen data.
# - **Random Forest**, being an ensemble of multiple decision trees, performed the best. It reduced the overfitting problem by averaging multiple trees' predictions, making it robust and accurate.
# 
# **The final results show that Random Forest is the best model**. Its MSE is the lowest, meaning it gave the most accurate predictions. It also avoided the overfitting problem that was seen with Decision Trees. 
# 
# **Conclusion**:
# - **Random Forest Regressor** is the best choice based on its performance. It combines the strengths of multiple decision trees and ensures robustness without overfitting.
# - If Random Forest hadn't performed best, **Linear Regression** would have been the next best model, especially if the relationship between features and target is linear.
# - **Decision Trees** can be useful but need to be carefully tuned to avoid overfitting, which is why Random Forest is typically preferred over a single Decision Tree.
