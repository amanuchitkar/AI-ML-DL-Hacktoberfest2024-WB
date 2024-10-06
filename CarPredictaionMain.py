import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import pickle

# Load and clean data
def clean_data(file_path):
    data = pd.read_csv(file_path)
    
    # Filter out rows where 'year' is non-numeric
    data = data[data['year'].str.isnumeric()]
    data['year'] = data['year'].astype(int)
    
    # Remove 'Ask For Price' rows and convert Price to integer
    data = data[data['Price'] != 'Ask For Price']
    data['Price'] = data['Price'].str.replace(',', '').astype(int)
    
    # Clean 'kms_driven' column
    data['kms_driven'] = data['kms_driven'].str.split(' ').str[0].str.replace(',', '')
    data = data[data['kms_driven'].str.isnumeric()]
    data['kms_driven'] = data['kms_driven'].astype(int)
    
    # Remove rows with missing fuel_type
    data = data[~data['fuel_type'].isna()]
    
    # Clean the 'name' column to include only the first three words
    data['name'] = data['name'].str.split(' ').str.slice(0, 3).str.join(' ')
    
    # Remove outliers based on Price
    data = data[data['Price'] < 6e6].reset_index(drop=True)
    
    return data

# Load cleaned data
data = clean_data('cleaned_data.csv')

# Feature matrix and target vector
X = data.drop(columns='Price')
y = data['Price']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=433)

# OneHotEncoder for categorical features
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

# Column transformer for handling categorical and numeric data
column_transformer = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

# Linear regression model pipeline
lr = LinearRegression()
pipe = make_pipeline(column_transformer, lr)

# Train the model
pipe.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipe.predict(X_test)
print(f"R2 score: {r2_score(y_test, y_pred)}")

# Saving the model using pickle
with open('LinearRegressionModel.pkl', 'wb') as model_file:
    pickle.dump(pipe, model_file)
