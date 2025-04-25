import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import mean_absolute_error,mean_squared_error, mean_absolute_percentage_error, r2_score

from statsmodels.tsa.stattools import acf
from datetime import datetime

# Import necessary classes from main program
from main_DM-FusionNet import (
    TimeConfig, PriceDataset, FusionModel, 
    collate_fn, evaluate_model,process_data, PriceDataset, FusionModel, 
        TimeConfig, TrainConfig, collate_fn
)

import os


def plot_diagnostics(model, test_loader, test_dataset, test_day_data, figsize=(15, 12)):
    """
    Create diagnostic plots for deep learning model, including autocorrelation analysis
    """
    model.eval()
    all_preds = []
    all_targets = []

    # Get predictions and true values
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            day_input, month_input, target = batch
            output, l2_reg, _, _ = model(day_input, month_input)
            all_preds.extend(output.squeeze().tolist())
            all_targets.extend(target.tolist())

    # Convert to numpy array and reshape
    preds = np.array(all_preds).reshape(-1, 1)
    targets = np.array(all_targets).reshape(-1, 1)

    # Inverse standardization - using updated dataset method
    preds = test_dataset.inverse_transform_price(preds)
    targets = test_dataset.inverse_transform_price(targets)

    # Calculate residuals
    residuals = preds - targets

    # Create subplots (2x2 layout)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Residuals over time plot (top left)
    test_dates = test_day_data['Date'].values[TimeConfig.DAY_LOOK_BACK:]
    axes[0, 0].plot(test_dates, residuals, '-o', alpha=0.5, markersize=2)
    axes[0, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0, 0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Residual histogram (top right)
    axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.7)
    mu, std = float(np.mean(residuals)), float(np.std(residuals))
    x = np.linspace(mu - 3 * std, mu + 3 * std, 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, std), 'r-', lw=2, label='Normal Dist.')
    axes[0, 1].set_xlabel('Residual Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].legend()

    # 3. QQ plot (bottom left)
    stats.probplot(residuals.ravel(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Normal Q-Q Plot')

    # 4. Autocorrelation plot (bottom right)
    nlags = 40
    acf_values = acf(residuals.ravel(), nlags=nlags)
    lags = np.arange(nlags + 1)
    confidence_interval = 1.96 / np.sqrt(len(residuals))  # 95% confidence interval

    axes[1, 1].bar(lags, acf_values, width=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='-')
    axes[1, 1].axhline(y=confidence_interval, color='r', linestyle='--', alpha=0.3)
    axes[1, 1].axhline(y=-confidence_interval, color='r', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Autocorrelation')
    axes[1, 1].set_title('Residual Autocorrelation (ACF)')

    plt.tight_layout()
    plt.savefig('img/model_diagnostics.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Return evaluation metrics for other purposes
    return float(r2_score(targets, preds))


def add_noise(data, noise_level):
    """
    Add Gaussian noise to data
    noise_level: Noise level (proportion of standard deviation)
    """
    std = data.std()
    noise = np.random.normal(0, std * noise_level, size=data.shape)
    return data + noise


def test_noise_robustness(model, test_day_data, month_data, train_dataset, 
                          noise_levels=[0.05, 0.10, 0.15]):
    """
    Test model robustness to noise
    """
    results = {}
    original_prices = test_day_data['Price'].values

    # Evaluate original performance using loaded test dataset
    original_dataset = PriceDataset(
        test_day_data.copy(), 
        month_data, 
        TimeConfig,
        scaler=train_dataset.get_scalers(), 
        is_train=False
    )
    
    original_loader = DataLoader(
        original_dataset, 
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )

    model.eval()
    original_metrics = evaluate_predictions(model, original_loader, original_dataset)
    results['original'] = original_metrics

    for noise_level in noise_levels:
        noisy_data = test_day_data.copy()
        # Save original prices before adding noise
        orig_prices = noisy_data['Price'].values.copy()
        # Add noise
        noisy_prices = add_noise(orig_prices, noise_level)
        # Replace price column
        noisy_data['Price'] = noisy_prices

        # Create noisy dataset
        noisy_dataset = PriceDataset(
            noisy_data, 
            month_data, 
            TimeConfig,
            scaler=train_dataset.get_scalers(), 
            is_train=False
        )
        
        noisy_loader = DataLoader(
            noisy_dataset, 
            batch_size=64,
            shuffle=False,
            collate_fn=collate_fn
        )

        noise_metrics = evaluate_predictions(model, noisy_loader, noisy_dataset)
        results[f'noise_{noise_level}'] = noise_metrics

    # Plot results
    plot_noise_results(results, test_day_data['Date'].values[TimeConfig.DAY_LOOK_BACK:], noise_levels)

    # Calculate stability metrics
    stability_metrics, avg_sensitivities = calculate_stability_metrics(results, noise_levels)

    # Print detailed report
    print("\n=== Noise Robustness Test Report ===")
    print("\nOriginal Data Performance:")
    print(f"RMSE: {results['original']['rmse']:.4f}")
    print(f"R²: {results['original']['r2']:.4f}")
    print(f"MAPE: {results['original']['mape']:.2%}")

    for noise_level in noise_levels:
        noise_key = f'noise_{noise_level}'
        print(f"\nNoise Level {noise_level * 100}% Performance:")
        print(f"RMSE: {results[noise_key]['rmse']:.4f} "
              f"(Change: {(results[noise_key]['rmse'] / results['original']['rmse'] - 1) * 100:.1f}%)")
        print(f"R²: {results[noise_key]['r2']:.4f} "
              f"(Change: {(results[noise_key]['r2'] / results['original']['r2'] - 1) * 100:.1f}%)")
        print(f"MAPE: {results[noise_key]['mape']:.2%} "
              f"(Change: {(results[noise_key]['mape'] / results['original']['mape'] - 1) * 100:.1f}%)")

    print("\nStability Metrics:")
    print(f"RMSE Average Sensitivity: {avg_sensitivities['avg_rmse_sensitivity']:.4f}")
    print(f"R² Average Sensitivity: {avg_sensitivities['avg_r2_sensitivity']:.4f}")
    print(f"MAPE Average Sensitivity: {avg_sensitivities['avg_mape_sensitivity']:.4f}")

    # Return all results, including stability metrics
    return {
        'predictions': results,
        'stability_metrics': stability_metrics,
        'avg_sensitivities': avg_sensitivities
    }


def evaluate_predictions(model, loader, dataset):
    """
    Evaluate model prediction results
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            day_input, month_input, target = batch
            output, l2_reg, day_weight, month_weight = model(day_input, month_input)

            all_preds.extend(output.squeeze().tolist())
            all_targets.extend(target.tolist())

    # Convert prediction results
    preds = np.array(all_preds).reshape(-1, 1)
    targets = np.array(all_targets).reshape(-1, 1)

    # Inverse standardization - using updated dataset method
    preds = dataset.inverse_transform_price(preds)
    targets = dataset.inverse_transform_price(targets)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    mape = mean_absolute_percentage_error(targets, preds)

    return {
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'predictions': preds.flatten(),
        'targets': targets.flatten()
    }


def plot_noise_results(results, dates, noise_levels):
    """
    Plot prediction results comparison under different noise levels
    """
    n_rows = len(noise_levels) + 1
    fig, axs = plt.subplots(n_rows, 1, figsize=(15, 5 * n_rows))

    # Plot prediction results for original data
    axs[0].plot(dates, results['original']['targets'], label='Actual', color='blue')
    axs[0].plot(dates, results['original']['predictions'], label='Predicted',
                color='red', linestyle='--')
    axs[0].set_title(f"Original Data\n"
                     f"RMSE: {results['original']['rmse']:.2f}, "
                     f"R²: {results['original']['r2']:.2f}, "
                     f"MAPE: {results['original']['mape']:.2%}")
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].legend()
    axs[0].grid(True)

    # Plot prediction results for different noise levels
    for i, noise_level in enumerate(noise_levels, 1):
        noise_key = f'noise_{noise_level}'

        axs[i].plot(dates, results[noise_key]['targets'], label='Actual (with noise)',
                    color='blue', alpha=0.6)
        axs[i].plot(dates, results[noise_key]['predictions'], label='Predicted',
                    color='red', linestyle='--')

        # Performance change percentage
        rmse_change = (results[noise_key]['rmse'] / results['original']['rmse'] - 1) * 100
        r2_change = (results[noise_key]['r2'] / results['original']['r2'] - 1) * 100
        mape_change = (results[noise_key]['mape'] / results['original']['mape'] - 1) * 100

        axs[i].set_title(f"Noise Level: {noise_level * 100}%\n"
                         f"RMSE: {results[noise_key]['rmse']:.2f} ({rmse_change:+.1f}%), "
                         f"R²: {results[noise_key]['r2']:.2f} ({r2_change:+.1f}%), "
                         f"MAPE: {results[noise_key]['mape']:.2%} ({mape_change:+.1f}%)")

        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig('img/noise_robustness.png', bbox_inches='tight', dpi=300)
    plt.close()


def calculate_stability_metrics(results, noise_levels):
    """
    Calculate model stability metrics for noise
    """
    stability_metrics = {
        'rmse_sensitivity': [],
        'r2_sensitivity': [],
        'mape_sensitivity': []
    }

    for noise_level in noise_levels:
        noise_key = f'noise_{noise_level}'

        # Calculate sensitivity of metrics to noise (change rate/noise level)
        rmse_change = (results[noise_key]['rmse'] / results['original']['rmse'] - 1) / noise_level
        r2_change = (results[noise_key]['r2'] / results['original']['r2'] - 1) / noise_level
        mape_change = (results[noise_key]['mape'] / results['original']['mape'] - 1) / noise_level

        stability_metrics['rmse_sensitivity'].append(rmse_change)
        stability_metrics['r2_sensitivity'].append(r2_change)
        stability_metrics['mape_sensitivity'].append(mape_change)

    # Calculate average sensitivity
    avg_sensitivities = {
        'avg_rmse_sensitivity': np.mean(np.abs(stability_metrics['rmse_sensitivity'])),
        'avg_r2_sensitivity': np.mean(np.abs(stability_metrics['r2_sensitivity'])),
        'avg_mape_sensitivity': np.mean(np.abs(stability_metrics['mape_sensitivity']))
    }

    return stability_metrics, avg_sensitivities


def save_test_predictions(model, test_loader, test_dataset, test_day_data):
    """
    Save test set true values and predictions to CSV file
    
    Parameters:
    model - Trained model
    test_loader  - Test data loader
    test_dataset - Test dataset
    test_day_data - Test date data
    
    Returns:
    None - Directly saves results to CSV file
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_dates = []
    all_day_weights = []
    all_month_weights = []
    
    # Get predictions and true values
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            day_input, month_input, target = batch
            output, _, day_weight, month_weight = model(day_input, month_input)
            
            all_preds.extend(output.squeeze().tolist())
            all_targets.extend(target.tolist())
            all_day_weights.extend(day_weight.tolist())
            all_month_weights.extend(month_weight.tolist())
    
    # Get corresponding dates
    dates = test_day_data['Date'].values[TimeConfig.DAY_LOOK_BACK:]
    if len(dates) > len(all_preds):
        dates = dates[:len(all_preds)]
    elif len(dates) < len(all_preds):
        # If dates are insufficient, truncate prediction results
        all_preds = all_preds[:len(dates)]
        all_targets = all_targets[:len(dates)]
        all_day_weights = all_day_weights[:len(dates)]
        all_month_weights = all_month_weights[:len(dates)]
    
    # Convert to numpy array and reshape
    preds = np.array(all_preds).reshape(-1, 1)
    targets = np.array(all_targets).reshape(-1, 1)
    
    # Inverse standardization
    preds = test_dataset.inverse_transform_price(preds)
    targets = test_dataset.inverse_transform_price(targets)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual': targets.flatten(),
        'Predicted': preds.flatten(),
        'Day_Weight': all_day_weights,
        'Month_Weight': all_month_weights
    })
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    mape = mean_absolute_percentage_error(targets, preds)
    
    # Add metrics information to results
    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R2', 'MAPE'],
        'Value': [rmse, mae, r2, mape]
    })
    
    # Ensure directory exists
    os.makedirs('results', exist_ok=True)
    
    # Save results
    results_df.to_csv('results/test_predictions.csv', index=False)
    metrics_df.to_csv('results/test_metrics.csv', index=False)
    
    print("Test results saved to 'results/test_predictions.csv'")
    print("Test metrics saved to 'results/test_metrics.csv'")
    
    return results_df

# Usage example
if __name__ == "__main__":
    
    # Create necessary directories
    os.makedirs('img', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load data
    day_data = pd.read_csv('data/day.csv')
    month_data = pd.read_csv('data/month.csv')
    
    # Data processing
    day_data, month_data = process_data(day_data, month_data)
    
    # Ensure data is sorted by time
    day_data = day_data.sort_values('Date')
    
    # Calculate dataset size
    total_size = len(day_data)
    train_size = int(total_size * TrainConfig.TRAIN_RATIO)
    val_size = int(total_size * TrainConfig.VAL_RATIO)
    
    # Split dataset by time order
    train_day = day_data[:train_size]
    val_day = day_data[train_size:train_size + val_size]
    test_day = day_data[train_size + val_size:]
    
    # Create datasets
    train_dataset = PriceDataset(train_day, month_data, TimeConfig, is_train=True)
    test_dataset = PriceDataset(test_day, month_data, TimeConfig,
                               scaler=train_dataset.get_scalers(), is_train=False)
    
    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=TrainConfig.BATCH_SIZE, 
                            shuffle=False, collate_fn=collate_fn)
    
    # Load model
    model = FusionModel()
    checkpoint = torch.load('models/best_model-0.8703.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Run diagnostics
    plot_diagnostics(model, test_loader, test_dataset, test_day)
    
    # Save test results
    save_test_predictions(model, test_loader, test_dataset, test_day)
    
    # Test noise robustness
    test_noise_robustness(model, test_day, month_data, train_dataset,
                         noise_levels=[0.05, 0.10, 0.15])