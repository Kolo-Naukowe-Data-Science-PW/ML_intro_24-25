# Load libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load data
data = pd.read_csv("https://raw.githubusercontent.com/Kolo-Naukowe-Data-Science-PW/ML_intro_24-25/refs/heads/main/data/Loan_Default.csv")

# Split data, drop unnecessary columns
y = data['Status']
x = data.drop(['Status', 'ID', 'year'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Extract numerical and categorical columns
categorical_columns = [col for col in X_train.columns if X_train[col].dtype == 'object']
numerical_columns = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]

# Create preprocessing pipelines

# Numerical pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both pipelines
pipeline = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Fit and transform data
X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# Convert back to DataFrame
X_train_transformed = pd.DataFrame(X_train_transformed)
X_test_transformed = pd.DataFrame(X_test_transformed)

# Add column names
X_train_transformed.columns = numerical_columns + list(pipeline.named_transformers_['cat']['onehot'].get_feature_names_out())
X_test_transformed.columns = numerical_columns + list(pipeline.named_transformers_['cat']['onehot'].get_feature_names_out())

# Save transformed data
X_train_transformed.to_csv('X_train.csv', index=False)
X_test_transformed.to_csv('X_test.csv', index=False)

# Save target data
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Save pipeline (optional, for transforming data in the future)
joblib.dump(pipeline, 'pipeline.pkl')
