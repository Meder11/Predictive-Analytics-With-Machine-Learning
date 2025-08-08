import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('Sales_data.csv')
print("Data loaded successfully.")

# Strip extra spaces from column names
data.columns = data.columns.str.strip()

# Handle missing values by filling with 0
data.fillna(0, inplace=True)
print("Missing values handled by filling with 0.")

# Check for duplicates and remove them
duplicates = data.duplicated().sum()
if duplicates > 0:
    data.drop_duplicates(inplace=True)
    print("Duplicate rows removed.")

# List of columns to clean
cols = ['Price', 'Sales']

# Iterate over the columns and clean them if they exist
for col in cols:
    if col in data.columns:  # Check if the column exists in the dataset
        # Remove dollar signs, commas, and extra spaces
        data[col] = data[col].replace({r'\$': '', r',': '', r'\s+': ''}, regex=True)
        # Convert the cleaned column to float
        data[col] = data[col].astype(float)
        print(f"{col} cleaned successfully.")
    else:
        print(f"Error: {col} not found in the dataset.")

# Convert the 'OrderDate' column to numeric features (year, month, day)
if 'OrderDate' in data.columns:
    data['OrderDate'] = pd.to_datetime(data['OrderDate'], errors='coerce')  # Convert to datetime
    data['Year'] = data['OrderDate'].dt.year
    data['Month'] = data['OrderDate'].dt.month
    data['Day'] = data['OrderDate'].dt.day
    data.drop(['OrderDate'], axis=1, inplace=True)  # Drop the original 'OrderDate' column
    print("'OrderDate' column cleaned and split into Year, Month, Day.")

# One-Hot Encoding for categorical variables (like 'Status')
data = pd.get_dummies(data, columns=['Status'], drop_first=True)  # Convert 'Status' to dummy variables
print("'Status' column encoded with One-Hot Encoding.")

# Drop any remaining non-numeric columns (like text columns)
data = data.select_dtypes(include=['float64', 'int64'])  # Keep only numeric columns
print("Non-numeric columns dropped.")

# Split the dataset into features and target variable
X = data.drop('Price', axis=1)  # Drop 'Price' column from features
y = data['Price']  # Target column 'Price'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully.")

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest Mean Squared Error: {mse_rf}")
print(f"Random Forest R-squared: {r2_rf}")
