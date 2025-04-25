import pandas as pd
import numpy as np

# Read data
df = pd.read_csv('data/prediction_results01.csv')
df['Date'] = pd.to_datetime(df['Date'])


def analyze_period(df, start_date, end_date):
    """Analyze the prediction performance in a specific period"""
    period = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    #Calculate basic error metrics
    mae = abs(period['Predicted'] - period['Ground_truth']).mean()
    mape = (abs(period['Predicted'] - period['Ground_truth']) / period['Ground_truth']).mean() * 100
    max_error = abs(period['Predicted'] - period['Ground_truth']).max()

    # Calculate the 5 - day trend prediction accuracy
    actual_change = period['Ground_truth'].diff(5)
    pred_change = period['Predicted'].diff(5)
    threshold = period['Ground_truth'] * 0.005
    significant_changes = actual_change.abs() > threshold
    correct_trends = (actual_change * pred_change > 0)
    trend_accuracy = (
            correct_trends[significant_changes].sum() /
            significant_changes.sum()
    ) if significant_changes.sum() > 0 else np.nan

    return {
        'samples': len(period),
        'mae': mae,
        'mape': mape,
        'max_error': max_error,
        'trend_accuracy': trend_accuracy,
        'price_range': [period['Ground_truth'].min(), period['Ground_truth'].max()]
    }


# Define each period
periods = [
    ('Low-level Oscillation Period', '2023-04-6', '2023-08-28'),
    ('Breakthrough Uptrend Period', '2023-08-28', '2023-10-17'),
    ('High-level Consolidation Period', '2023-10-17', '2023-12-12'),
    (Adjustment Period', '2023-12-12', '2024-02-23'),
    ('Stabilization and Recovery Period', '2024-02-23', '2024-06-28'),
]
# Analyze the prediction performance in each period
print("\nPrediction performance in different market phases：")
for period_name, start_date, end_date in periods:
    results = analyze_period(df, start_date, end_date)
    print(f"\n{period_name}:")
    print(f"Number of samples: {results['samples']}")
    print(f"Price range: {results['price_range'][0]:.2f}-{results['price_range'][1]:.2f}")
    print(f"MAE: {results['mae']:.2f}")
    print(f"MAPE: {results['mape']:.2f}%")
    print(f"Maximum error: {results['max_error']:.2f}")
    print(f"Trend prediction accuracy: {results['trend_accuracy']:.4f}")