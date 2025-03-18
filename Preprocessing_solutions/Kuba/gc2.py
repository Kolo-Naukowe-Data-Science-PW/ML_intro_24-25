#!/usr/bin/env python3

import joblib
import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    try:
        logger.info(f"Loading pipeline from {args.pipeline_path}")
        pipeline = joblib.load(args.pipeline_path)
        
        logger.info(f"Loading data from {args.data_path}")
        new_data = pd.read_csv(args.data_path)
        
        logger.info("Applying pipeline to new data")
        if args.target_column in new_data.columns:
            X_new = new_data.drop(columns=[args.target_column])
            y_new = new_data[args.target_column]
            
            predictions = pipeline.predict(X_new)
            logger.info(f"Made {len(predictions)} predictions")
            
            if hasattr(pipeline, 'predict_proba'):
                probabilities = pipeline.predict_proba(X_new)
                results = pd.DataFrame({
                    'actual': y_new,
                    'predicted': predictions
                })
                
                for i in range(probabilities.shape[1]):
                    results[f'prob_class_{i}'] = probabilities[:, i]
            else:
                results = pd.DataFrame({
                    'actual': y_new,
                    'predicted': predictions
                })
        else:
            predictions = pipeline.predict(new_data)
            results = pd.DataFrame({'predicted': predictions})
        
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