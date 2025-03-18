#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class CustomFeatureTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, drop_original=False, cap_outliers=True, 
                 outlier_threshold=3.0, create_interactions=False):
        self.drop_original = drop_original
        self.cap_outliers = cap_outliers
        self.outlier_threshold = outlier_threshold
        self.create_interactions = create_interactions
        self.feature_stats_ = {}
        
    def fit(self, X, y=None):
        for col in X.select_dtypes(include=['number']).columns:
            self.feature_stats_[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'min': X[col].min(),
                'max': X[col].max(),
                'median': X[col].median(),
                'upper_cap': X[col].mean() + (X[col].std() * self.outlier_threshold),
                'lower_cap': X[col].mean() - (X[col].std() * self.outlier_threshold)
            }
        
        logger.info(f"Fitted CustomFeatureTransformer with {len(self.feature_stats_)} numerical features")
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if self.cap_outliers:
            for col, stats in self.feature_stats_.items():
                if col in X_copy.columns:
                    X_copy[col] = X_copy[col].clip(
                        lower=stats['lower_cap'], 
                        upper=stats['upper_cap']
                    )
            logger.info("Applied outlier capping")
        
        numerical_cols = X_copy.select_dtypes(include=['number']).columns
        
        if 'loan_amount' in numerical_cols and 'income' in numerical_cols:
            X_copy['loan_to_income'] = X_copy['loan_amount'] / (X_copy['income'] + 1)
            logger.info("Created loan_to_income feature")
            
        if 'rate_of_interest' in numerical_cols and 'loan_amount' in numerical_cols:
            X_copy['interest_burden'] = X_copy['rate_of_interest'] * X_copy['loan_amount'] / 100
            logger.info("Created interest_burden feature")
        
        if 'Credit_Score' in numerical_cols and 'LTV' in numerical_cols:
            X_copy['risk_indicator'] = X_copy['LTV'] / (X_copy['Credit_Score'] + 1) * 100
            logger.info("Created risk_indicator feature")
            
        if self.create_interactions:
            num_features = [col for col in numerical_cols 
                          if col in X_copy.columns 
                          and col not in ['ID', 'year']]
            
            for i, col1 in enumerate(num_features):
                for col2 in num_features[i+1:]:
                    X_copy[f'{col1}_x_{col2}'] = X_copy[col1] * X_copy[col2]
            
            logger.info(f"Created interaction terms between numerical features")
        
        return X_copy

def load_and_explore_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dane wczytane z {file_path}")
        
        logger.info(f"Typy danych:\n{df.dtypes}")
        logger.info(f"Statystyki opisowe:\n{df.describe(include='all')}")
        logger.info(f"Liczba brakujących wartości:\n{df.isnull().sum()}")
        
        plt.figure(figsize=(15, 10))
        
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns[:6]
        for i, col in enumerate(num_cols):
            plt.subplot(2, 3, i+1)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
        plt.savefig('num_distribution.png')
        
        plt.figure(figsize=(15, 10))
        cat_cols = df.select_dtypes(include=['object']).columns[:6]
        for i, col in enumerate(cat_cols):
            plt.subplot(2, 3, i+1)
            top_cats = df[col].value_counts().head(10)
            sns.barplot(x=top_cats.index, y=top_cats.values)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Count of {col}')
            plt.tight_layout()
        plt.savefig('cat_count.png')
        
        logger.info(f"Wizualizacje zapisane jako 'num_distribution.png' oraz 'cat_count.png'")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def identify_column_types(df, target_column):
    features_df = df.drop(columns=[target_column], errors='ignore')
    
    numerical_columns = features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Kolumny numeryczne: {numerical_columns}")
    logger.info(f"Kolumny kategoryczne: {categorical_columns}")
    
    return numerical_columns, categorical_columns

def create_preprocessing_pipeline(numerical_columns, categorical_columns):
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_columns),
            ('cat', categorical_pipeline, categorical_columns)
        ],
        remainder='drop'
    )
    
    logger.info("Created preprocessing pipeline")
    return preprocessor

def train_and_evaluate_pipeline(df, preprocessor, custom_transformer, target_column, 
                              model_type='random_forest', handle_imbalance=False):
    if target_column not in df.columns:
        error_msg = f"Target column '{target_column}' not found in dataframe. Available columns: {df.columns.tolist()}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    logger.info(f"Split data: X_train={X_train.shape}, X_test={X_test.shape}")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    else:
        model = LogisticRegression(random_state=42, max_iter=1000)
    
    if handle_imbalance:
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_train_transformed = custom_transformer.fit_transform(X_train)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
        logger.info(f"Applied SMOTE: before={dict(zip(*np.unique(y_train, return_counts=True)))}, "
                   f"after={dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")
        
        final_pipeline = Pipeline([
            ('model', model)
        ])
        
        final_pipeline.fit(X_train_resampled, y_train_resampled)
        
        X_test_preprocessed = preprocessor.transform(X_test)
        
        y_pred = final_pipeline.predict(X_test_preprocessed)
    else:
        full_pipeline = Pipeline([
            ('features', custom_transformer),
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        full_pipeline.fit(X_train, y_train)
        
        y_pred = full_pipeline.predict(X_test)
        
        final_pipeline = full_pipeline
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    logger.info(f"Model Evaluation:\n"
               f"Accuracy: {accuracy:.4f}\n"
               f"Precision: {precision:.4f}\n"
               f"Recall: {recall:.4f}\n"
               f"F1 Score: {f1:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    logger.info("Confusion matrix saved as 'confusion_matrix.png'")
    
    report = classification_report(y_test, y_pred)
    logger.info(f"Classification Report:\n{report}")
    
    X_test_processed = preprocessor.transform(custom_transformer.transform(X_test))
    if isinstance(X_test_processed, np.ndarray):
        pd.DataFrame(X_test_processed).to_csv('processed_test_data.csv', index=False)
    else:
        X_test_processed.to_csv('processed_test_data.csv', index=False)
    
    logger.info("Processed test data saved to 'processed_test_data.csv'")
    
    return final_pipeline, {
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1,
        'y_test': y_test,
        'y_pred': y_pred
    }

def save_and_reload_pipeline(pipeline, file_path):
    joblib.dump(pipeline, file_path)
    logger.info(f"Pipeline saved to {file_path}")
    
    reloaded_pipeline = joblib.load(file_path)
    logger.info(f"Pipeline reloaded from {file_path}")
    
    return reloaded_pipeline

def tune_hyperparameters(pipeline, X_train, y_train, param_grid):
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search

def apply_dimensionality_reduction(X, n_components=2, plot=True):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    explained_variance = pca.explained_variance_ratio_
    logger.info(f"Explained variance by component: {explained_variance}")
    logger.info(f"Total explained variance: {sum(explained_variance):.4f}")
    
    if plot and n_components >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
        plt.ylabel(f'PC2 ({explained_variance[1]::.2%})')
        plt.title('PCA of the dataset')
        plt.savefig('pca_visualization.png')
        logger.info("PCA visualization saved as 'pca_visualization.png'")
    
    return X_pca

def main(args):
    df = load_and_explore_data(args.data_path)
    
    numerical_columns, categorical_columns = identify_column_types(df, args.target_column)
    
    preprocessor = create_preprocessing_pipeline(numerical_columns, categorical_columns)
    
    custom_transformer = CustomFeatureTransformer(
        cap_outliers=args.cap_outliers,
        create_interactions=args.create_interactions
    )
    
    pipeline_model = train_and_evaluate_pipeline(df, preprocessor, custom_transformer,
                                             args.target_column, args.model_type, 
                                             args.handle_imbalance)
    
    save_path = os.path.join(args.output_dir, f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib')
    saved_pipeline = save_and_reload_pipeline(pipeline_model[0], save_path)
    
    if args.tune_hyperparameters:
        param_grid = {
            'model__n_estimators': [50, 100, 200] if args.model_type == 'random_forest' else None,
            'model__max_depth': [None, 10, 20, 30] if args.model_type == 'random_forest' else None,
            'model__C': [0.1, 1, 10] if args.model_type == 'logistic' else None
        }
        param_grid = {k: v for k, v in param_grid.items() if v is not None}
        
        if param_grid:
            X = df.drop(columns=[args.target_column])
            y = df[args.target_column]
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            
            tuned_model = tune_hyperparameters(pipeline_model[0], X_train, y_train, param_grid)
            
            best_model_path = os.path.join(args.output_dir, 'best_pipeline.joblib')
            joblib.dump(tuned_model.best_estimator_, best_model_path)
            logger.info(f"Best model saved to {best_model_path}")
    
    if args.apply_pca:
        X = df.drop(columns=[args.target_column])
        X_preprocessed = preprocessor.fit_transform(custom_transformer.fit_transform(X))
        
        X_pca = apply_dimensionality_reduction(X_preprocessed, n_components=2)
        
        pd.DataFrame(X_pca, columns=['PC1', 'PC2']).to_csv('pca_results.csv', index=False)
        logger.info("PCA results saved to 'pca_results.csv'")
    
    logger.info("Pipeline execution completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Data Processing Pipeline")
    
    parser.add_argument("--data_path", type=str, default="../../data/Loan_Default.csv",
                      help="Path to the input CSV file")
    parser.add_argument("--target_column", type=str, required=True,
                      help="Name of the target column")
    parser.add_argument("--model_type", type=str, choices=['random_forest', 'logistic'], 
                      default='random_forest', help="Type of model to use")
    parser.add_argument("--output_dir", type=str, default="./models",
                      help="Directory to save model artifacts")
    parser.add_argument("--cap_outliers", action="store_true", 
                      help="Whether to cap outliers in numerical features")
    parser.add_argument("--create_interactions", action="store_true",
                      help="Whether to create interaction terms between features")
    parser.add_argument("--handle_imbalance", action="store_true",
                      help="Whether to handle class imbalance using SMOTE")
    parser.add_argument("--tune_hyperparameters", action="store_true",
                      help="Whether to perform hyperparameter tuning")
    parser.add_argument("--apply_pca", action="store_true",
                      help="Whether to apply PCA for dimensionality reduction")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)