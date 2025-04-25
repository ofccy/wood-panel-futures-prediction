import pandas as pd
import numpy as np

# Read data
df = pd.read_csv('data/prediction_results01.csv')
df['Date'] = pd.to_datetime(df['Date'])


def calc_trend_accuracy(df, window):
    """Calculate the accuracy of trend prediction using the difference method"""
    # Calculate price changes
    actual_change = df['Ground_truth'].diff(window)
    pred_change = df['Predicted'].diff(window)

    # Set the threshold
    threshold = df['Ground_truth'] * 0.005

    # Accuracy of trend prediction
    significant_changes = actual_change.abs() > threshold
    correct_trends = (actual_change * pred_change > 0)
    trend_accuracy = (
            correct_trends[significant_changes].sum() /
            significant_changes.sum()
    )

    return trend_accuracy


def calc_error_metrics(df, window):
    """  Calculate MAE and MAPE using the moving average method"""
    #Calculate the moving average
    actual_ma = df['Ground_truth'].rolling(window).mean()
    pred_ma = df['Predicted'].rolling(window).mean()

    # MAE
    mae = abs(pred_ma - actual_ma).mean()

    # MAPE
    mape = (abs(pred_ma - actual_ma) / actual_ma).mean() * 100

    return mae, mape


# Define time windows
short_term = [5, 10]  #  Short - term: 5 days and 10 days
mid_term = [20, 60]  #  Mid - term: 20 days and 60 days
long_term = [90]  # Long - term: 90 days

print("\n1. Short - term prediction performance：")
for window in short_term:
    acc = calc_trend_accuracy(df, window)
    mae, mape = calc_error_metrics(df, window)
    print(f"\n{window}day prediction：")
    print(f"Trend prediction accuracy: {acc:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

print("\n2. Mid - term prediction performance：")
for window in mid_term:
    acc = calc_trend_accuracy(df, window)
    mae, mape = calc_error_metrics(df, window)
    print(f"\n{window}day prediction：")
    print(f"Trend prediction accuracy: {acc:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

print("\n3. Long - term prediction performance：")
for window in long_term:
    acc = calc_trend_accuracy(df, window)
    mae, mape = calc_error_metrics(df, window)
    print(f"\n{window}day prediction：")
    print(f"Trend prediction accuracy: {acc:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")