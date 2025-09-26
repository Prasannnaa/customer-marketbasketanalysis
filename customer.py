import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


df = pd.read_csv("online Retail.csv", encoding='ISO-8859-1')
print("Dataset loaded successfully!")
print(df.head())
print(df.info())


df.dropna(subset=['CustomerID'], inplace=True)


df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]


df['Revenue'] = df['Quantity'] * df['UnitPrice']


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H.%M')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day


df['Country'] = df['Country'].astype('category').cat.codes

X = df[['Quantity', 'UnitPrice', 'Country', 'Month', 'Day']]
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

joblib.dump(model, "online_retail_regression_model.pkl")
print("Model saved successfully!")


# Example: Quantity=10, UnitPrice=20, Country=5, Month=12, Day=5
#new_data = [[10, 20, 5, 12, 5]]
#prediction = model.predict(new_data)
#print("Predicted Revenue for new data:", prediction[0])

import pandas as pd

new_data = pd.DataFrame([[10, 20, 5, 12, 5]], columns=['Quantity', 'UnitPrice', 'Country', 'Month', 'Day'])
prediction = model.predict(new_data)
print(prediction)