# ICMP Single Rule Behavior Detection

## Overview

This project implements a Machine Learning-based behavior detection system for ICMP network traffic.

It uses a pre-trained Machine Learning model and behavioral feature engineering to analyze network events and generate predictions for each event.

The prediction script processes network traffic data from a CSV file, extracts time-based behavioral features, and uses the trained model to classify the events.

## Project Files

| File | Description |
|---|---|
| `icmp_single_rule_behavior_model_local.joblib` | Pre-trained Machine Learning model and feature configuration |
| `predict_icmp_single_rule_fixed_v2.py` | Python script for feature engineering and prediction |
| `requirements.txt` | Required Python libraries |

## Features

- Loads a pre-trained Machine Learning model.
- Supports Kibana-style and ISO timestamp formats.
- Extracts time-based behavioral features.
- Analyzes source IP and source-destination communication patterns.
- Calculates event counts over 5, 10, 30, and 60-second windows.
- Generates predictions for network events.
- Provides prediction confidence and class probabilities when supported by the model.
- Exports prediction results to a CSV file.

## Requirements

- Python 3.10 or later
- joblib
- NumPy
- Pandas
- The Python package required by the serialized model estimator

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

Run the prediction script from the project directory:

```bash
python predict_icmp_single_rule_fixed_v2.py --model icmp_single_rule_behavior_model_local.joblib --csv input.csv --output predictions.csv
```

### Arguments

| Argument | Description |
|---|---|
| `--model` | Path to the trained model |
| `--csv` | Path to the input network traffic CSV file |
| `--output` | Path where prediction results will be saved |

## Input Data

The input CSV should contain:

- `@timestamp` or `timestamp`
- `source.ip`
- `destination.ip`

Additional columns may be required depending on the features stored in the trained model.

## Output

The script generates a CSV file containing:

- Original input data.
- Extracted behavioral features.
- `predicted_label`: Model prediction for each event.
- `confidence`: Maximum predicted class probability, when supported.
- Individual class probabilities, when supported.

## How It Works

1. Load the trained model and its required feature columns.
2. Read the input network traffic CSV file.
3. Parse and sort event timestamps.
4. Extract behavioral features based on recent network activity.
5. Prepare the feature matrix.
6. Generate predictions using the trained model.
7. Export the results to a CSV file.

## Notes

- The model file and prediction script must be accessible when running the application.
- Input timestamps must be in a supported format.
- The model package must contain `model` and `feature_cols` entries.
- The input data must provide the columns required by the trained model.
## Author

Machine Learning Project – ICMP Behavior Detection
