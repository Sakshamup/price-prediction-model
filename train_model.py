import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv(r"D:\practice\Clean_Dataset.csv")
data.dropna(inplace=True)

data.drop(['Unnamed: 0', 'flight'], axis=1, inplace=True)

X = data.drop('price', axis=1)
Y = data['price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

transformer = ColumnTransformer(transformers=[
    ('tf1', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
     ['source_city', 'departure_time', 'arrival_time', 'destination_city', 'airline']),
    ('tf2', OrdinalEncoder(categories=[['Economy', 'Business']]), ['class']),
    ('tf3', OrdinalEncoder(categories=[['zero', 'one', 'two_or_more']]), ['stops'])
], remainder='passthrough')

X_train_trans = transformer.fit_transform(X_train)
X_test_trans = transformer.transform(X_test)

model = xgb.XGBRegressor(objective='reg:squarederror', use_label_encoder=False)
model.fit(X_train_trans, Y_train)

Y_pred = model.predict(X_test_trans)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)
print(f'RMSE: {rmse:.2f}')
print(f'R² Score: {r2:.2f}')

joblib.dump(model, "xgboost_flight_model.pkl")
joblib.dump(transformer, "transformer.pkl")

print("Model and transformer saved successfully!")