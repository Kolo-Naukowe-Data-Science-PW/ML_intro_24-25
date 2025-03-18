#!/usr/bin/env python3
"""
Example script to demonstrate how to use a saved preprocessing pipeline
"""

import joblib
import pandas as pd
import argparse
import logging
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        
        numerical_cols = X_copy.select_dtypes(include=['number']).columns
        
        if 'loan_amount' in numerical_cols and 'income' in numerical_cols:
            X_copy['loan_to_income'] = X_copy['loan_amount'] / (X_copy['income'] + 1)
            
        if 'rate_of_interest' in numerical_cols and 'loan_amount' in numerical_cols:
            X_copy['interest_burden'] = X_copy['rate_of_interest'] * X_copy['loan_amount'] / 100
        
        if 'Credit_Score' in numerical_cols and 'LTV' in numerical_cols:
            X_copy['risk_indicator'] = X_copy['LTV'] / (X_copy['Credit_Score'] + 1) * 100
            
        if self.create_interactions:
            num_features = [col for col in numerical_cols 
                          if col in X_copy.columns 
                          and col not in ['ID', 'year']]
            
            for i, col1 in enumerate(num_features):
                for col2 in num_features[i+1:]:
                    X_copy[f'{col1}_x_{col2}'] = X_copy[col1] * X_copy[col2]
        
        return X_copy

def main(args):
    """Load a saved pipeline and apply it to new data"""
    try:
        # Load the pipeline
        logger.info(f"Loading pipeline from {args.pipeline_path}")
        pipeline = joblib.load(args.pipeline_path)
        
        # Load new data
        logger.info(f"Loading data from {args.data_path}")
        new_data = pd.read_csv(args.data_path)
        
        # Apply the pipeline to new data
        logger.info("Applying pipeline to new data")
        if args.target_column in new_data.columns:
            # If target column exists, separate it
            X_new = new_data.drop(columns=[args.target_column])
            y_new = new_data[args.target_column]
            
            # Make predictions
            predictions = pipeline.predict(X_new)
            logger.info(f"Made {len(predictions)} predictions")
            
            # If this is a classification task, add probabilities
            if hasattr(pipeline, 'predict_proba'):
                probabilities = pipeline.predict_proba(X_new)
                # Create a results DataFrame with predictions and probabilities
                results = pd.DataFrame({
                    'actual': y_new,
                    'predicted': predictions
                })
                
                # Add probability columns for each class
                for i in range(probabilities.shape[1]):
                    results[f'prob_class_{i}'] = probabilities[:, i]
            else:
                # For regression tasks
                results = pd.DataFrame({
                    'actual': y_new,
                    'predicted': predictions
                })
        else:
            # No target column, just make predictions
            predictions = pipeline.predict(new_data)
            results = pd.DataFrame({'predicted': predictions})
        
        # Save results
        results.to_csv(args.output_path, index=False)
        logger.info(f"Results saved to {args.output_path}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a saved pipeline to new data")
    
    parser.add_argument("--pipeline_path", type=str, required=True,
                     help="Path to the saved pipeline file (.joblib)")
    parser.add_argument("--data_path", type=str, required=True,
                     help="Path to the new data CSV file")
    parser.add_argument("--target_column", type=str, default=None,
                     help="Name of the target column if present in the new data")
    parser.add_argument("--output_path", type=str, default="pipeline_results.csv",
                     help="Path to save the predictions")
    
    args = parser.parse_args()
    main(args)