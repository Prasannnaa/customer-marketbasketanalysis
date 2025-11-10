Online Retail Revenue Prediction – Machine Learning Project

This project builds a Machine Learning model to predict the Revenue generated from online retail transactions using the popular Online Retail dataset.
It showcases end-to-end data cleaning, feature engineering, regression modeling, and prediction using Python and Scikit-Learn.

✅ Project Overview

The project focuses on:

Cleaning and preprocessing retail transaction data

Removing invalid quantities, missing IDs, and duplicates

Extracting useful features such as Month and Day

Encoding categorical values (Country)

Training a Random Forest Regression Model

Evaluating performance using MSE and R² Score

Saving the model with joblib

Predicting revenue for new unseen inputs

This project demonstrates fundamental Data Science skills suitable for internship-level evaluation.
✅ Technologies Used

Python

Pandas

Scikit-Learn

Joblib

Jupyter Notebook / Script

✅ Key Steps in the Project

1. Data Cleaning

Removed missing CustomerID entries

Filtered out negative or zero Quantity and UnitPrice

Converted InvoiceDate into datetime format

2. Feature Engineering

Created new features:

Revenue = Quantity × UnitPrice

Extracted Month and Day from InvoiceDate

Encoded Country as numeric values

3. Model Training

Model used:

RandomForestRegressor(n_estimators=100, random_state=42)

Evaluation metrics:

Mean Squared Error (MSE)

R² Score
4. Saving the Model

joblib.dump(model, "online_retail_regression_model.pkl")

5. Predicting New Values

new_data = pd.DataFrame([[10, 20, 5, 12, 5]],
                        columns=['Quantity', 'UnitPrice', 'Country', 'Month', 'Day'])
prediction = model.predict(new_data)
print(prediction)
✅ Project Structure
📁 online-retail-revenue-prediction/
│── customer.py
│── online_retail.csv
│── README.md
✅ Future Improvements

Add data visualizations (EDA)
Convert project into a Jupyter Notebook
Tune hyperparameters for better accuracy
Deploy model with Flask or FastAPI
✅ Author

Vadapalli Prasanna Lakshmi
B.Tech CSE | Data Science Enthusiast
Passionate about Python, ML, and building real projects.
