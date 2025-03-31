import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor

# Load dataset
data = pd.read_csv(r'D:\practice\Housing.csv')

# Handle categorical variables
data['furnishingstatus'] = data['furnishingstatus'].replace(
    {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
).astype(int)

# Encode categorical binary columns properly
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
encoder_dict = {}  # Store encoders for later use

for col in binary_cols:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    encoder_dict[col] = encoder

# Save the encoders
joblib.dump(encoder_dict, 'encoders.pkl')

# Define features and target variable
X = data.drop(columns='price', axis=1)
Y = data['price']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=2)

# Train CatBoost model with optimized parameters
cat_model = CatBoostRegressor(
    iterations=800,      # Increased iterations for better learning
    learning_rate=0.03,  # Lower learning rate for stable training
    depth=7,             # Slightly increased depth
    l2_leaf_reg=3,       # L2 regularization to prevent overfitting
    verbose=100
)
cat_model.fit(X_train, Y_train)

# Predictions
y_pred = cat_model.predict(X_test)

# Model evaluation
r2 = r2_score(Y_test, y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))

print(f"CatBoost R² Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Save model
joblib.dump(cat_model, 'catboost_model.pkl')

# Save training performance logs
with open("training_log.txt", "w") as f:
    f.write(f"CatBoost R² Score: {r2:.4f}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")

# Feature Importance Analysis
feature_importance = cat_model.get_feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Save feature importance
importance_df.to_csv("feature_importance.csv", index=False)

print("Model, encoders, and feature importance saved successfully!")
