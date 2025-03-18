
```sh
python gc1_copy.py --target_column Status

```

```sh
python gc2_copy.py --pipeline_path models/pipeline_20250311_183353.joblib \
                  --data_path ../../data/Loan_Default.csv \
                  --target_column Status \
                  --output_path predictions.csv
```