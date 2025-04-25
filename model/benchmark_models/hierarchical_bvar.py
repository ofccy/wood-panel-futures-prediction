import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import invgamma, norm
from scipy.stats import gaussian_kde

class HierarchicalBVAR:
    def __init__(self, look_back=20):
        """Initialize hierarchical prior BVAR model"""
        self.look_back = look_back
        self.scaler = StandardScaler()
        self.params = None
        self.hyperpriors = {
            'global_shrinkage': 0.01,   # Stronger global shrinkage
            'lag_decay': 0.12,          # Faster decay
        }
        
    def prepare_data(self, prices, features, i, is_training=True):
        """Prepare training data"""
        try:
            window_start = max(0, i - self.look_back)
            window_end = i
            
            # Get price data
            window_prices = prices[window_start:window_end]
            
            # Check data length
            if len(window_prices) < self.look_back:
                print(f"Data length insufficient: {len(window_prices)} < {self.look_back}")
                return pd.DataFrame()
            
            # Basic price features
            price_series = pd.Series(window_prices).ffill()  # Forward fill first
            price_returns = price_series.pct_change()
            
            # Get monthly feature data
            window_features = features.iloc[window_start:window_end].copy()
            
            # Check feature data length
            if len(window_features) != len(window_prices):
                print(f"Feature data length mismatch: {len(window_features)} != {len(window_prices)}")
                return pd.DataFrame()
            
            # Construct basic data dictionary
            data_dict = {
                'price': price_series,
                'price_return': price_returns
            }
            
            # Add monthly features
            for col in features.columns:
                feat_series = window_features[col].ffill()  # Forward fill first
                data_dict[col] = feat_series
                
                # Add monthly feature change rates
                feat_returns = feat_series.pct_change()  # Calculate change rate on filled data
                data_dict[f'{col}_change'] = feat_returns
            
            # Convert to DataFrame
            data = pd.DataFrame(data_dict)
            
            # Handle missing and infinite values
            data = data.replace([np.inf, -np.inf], np.nan)  # Handle infinite values first
            data = data.ffill()  # Forward fill
            data = data.bfill()  # Backward fill
            
            # Check data completeness
            if data.isnull().any().any():
                print("Missing values in data, columns:")
                print(data.columns[data.isnull().any()].tolist())
                return pd.DataFrame()
            
            return data
            
        except Exception as e:
            print(f"Data preparation error: {str(e)}")
            return pd.DataFrame()
def fit(self, prices, features):
        """Train hierarchical Bayesian VAR model"""
        try:
            i = len(prices)
            data = self.prepare_data(prices, features, i, is_training=True)
            
            if data.empty:
                raise ValueError("Data is empty")
            
            # Standardize data
            scaled_data = pd.DataFrame(
                self.scaler.fit_transform(data),
                columns=data.columns
            )
            
            # Calculate feature correlations
            corr_matrix = scaled_data.corr()['price'].abs().sort_values(ascending=False)
            print("\nFeature correlations:")
            print(corr_matrix)
            
            # Prepare training data
            X = scaled_data.values[:-1]
            y = scaled_data.values[1:, scaled_data.columns.get_loc('price')]
            
            # Set hierarchical priors
            n_vars = X.shape[1]
            
            # First layer: global hyperparameters
            global_scale = invgamma.rvs(a=15.0, scale=0.035)  # Stronger shape parameter control
            
            # Second layer: feature group hyperparameters
            price_corr = corr_matrix['price']
            price_return_corr = corr_matrix['price_return']
            monthly_corrs = corr_matrix[features.columns]
            
            # Improved dynamic group scale
            group_scales = {
                'price': invgamma.rvs(a=10.0, scale=0.005),  # Stronger price control
                'price_return': invgamma.rvs(a=7.0, scale=0.008 * price_return_corr),
                'monthly_very_high': invgamma.rvs(a=6.0, scale=0.025),  # Extremely high correlation monthly indicators
                'monthly_high': invgamma.rvs(a=5.5, scale=0.035),      # High correlation monthly indicators
                'monthly_medium': invgamma.rvs(a=5.0, scale=0.045),    # Medium correlation monthly indicators
                'monthly_low': invgamma.rvs(a=4.5, scale=0.055)        # Low correlation monthly indicators
            }
            
            # Third layer: improved individual parameter priors
            param_priors = {}
            for j, col in enumerate(scaled_data.columns):
                corr = corr_matrix.get(col, 0)
                
                if col == 'price':
                    group = 'price'
                    scale = group_scales[group] * 0.006
                    mean = 0.998  # Stronger autocorrelation assumption
                elif col == 'price_return':
                    group = 'price_return'
                    scale = group_scales[group]
                    mean = 0.0
                else:
                    # Improved monthly indicator grouping
                    if corr > 0.35:  # Raised threshold
                        group = 'monthly_very_high'
                        importance = np.tanh(7 * corr)  # Increase nonlinearity
                        scale = group_scales[group] * (0.03 + 0.06 * importance)
                        mean = 0.03 * importance
                    elif corr > 0.25:
                        group = 'monthly_high'
                        importance = np.tanh(6 * corr)
                        scale = group_scales[group] * (0.04 + 0.08 * importance)
                        mean = 0.025 * importance
                    elif corr > 0.15:
                        group = 'monthly_medium'
                        importance = np.tanh(5 * corr)
                        scale = group_scales[group] * (0.05 + 0.1 * importance)
                        mean = 0.02 * importance
                    else:
                        group = 'monthly_low'
                        importance = np.tanh(4 * corr)
                        scale = group_scales[group] * (0.06 + 0.12 * importance)
                        mean = 0.0
                
                param_priors[col] = {
                    'mean': mean,
                    'scale': scale * global_scale
                }
            
            # Improved time dependency - seven-layer time scale
            time_base = np.arange(self.look_back)
            time_weights = (
                0.30 * np.exp(-time_base * 0.5) +     # Current day (1 day)
                0.25 * np.exp(-time_base * 0.25) +    # Ultra-short term (2 days)
                0.20 * np.exp(-time_base * 0.15) +    # Short term (3-4 days)
                0.12 * np.exp(-time_base * 0.08) +    # Medium-short term (5-7 days)
                0.08 * np.exp(-time_base * 0.04) +    # Medium term (8-12 days)
                0.03 * np.exp(-time_base * 0.02) +    # Medium-long term (13-16 days)
                0.02 * np.exp(-time_base * 0.01)      # Long term (>16 days)
            )
            
            # Optimize prior precision matrix
            prior_precision = np.zeros((n_vars, n_vars))
            for i, col in enumerate(scaled_data.columns):
                base_precision = 1.0 / param_priors[col]['scale']
                
                if col == 'price':
                    # Multi-scale time weights for price autoregressive terms
                    day1 = np.mean(time_weights[:1])      # 1 day
                    day2_3 = np.mean(time_weights[1:3])   # 2-3 days
                    day4_6 = np.mean(time_weights[3:6])   # 4-6 days
                    day7_10 = np.mean(time_weights[6:10]) # 7-10 days
                    day11_15 = np.mean(time_weights[10:15]) # 11-15 days
                    day15_plus = np.mean(time_weights[15:])  # >15 days
                    
                    weighted_precision = base_precision * (
                        0.35 * day1 +
                        0.25 * day2_3 +
                        0.18 * day4_6 +
                        0.12 * day7_10 +
                        0.07 * day11_15 +
                        0.03 * day15_plus
                    )
                elif 'price_return' in col:
                    # Returns use short-term weights
                    weighted_precision = base_precision * (
                        0.6 * np.mean(time_weights[:3]) +  # Near 3 days
                        0.4 * np.mean(time_weights[3:7])   # 4-7 days
                    )
                else:
                    # Monthly indicators use correlation and time combined weights
                    corr_weight = np.abs(corr_matrix[col])
                    time_weight = np.mean(time_weights[:10])  # Use average of first 10 days
                    weighted_precision = base_precision * (
                        0.7 * (0.9 + 0.1 * corr_weight) +
                        0.3 * time_weight
                    )
                
                prior_precision[i, i] = weighted_precision
            
            # Calculate posterior parameters
            try:
                # Add small regularization term
                reg_matrix = 1e-6 * np.eye(n_vars)
                
                # Posterior covariance
                post_cov = np.linalg.inv(prior_precision + X.T @ X + reg_matrix)
                
                # Posterior mean (considering prior mean)
                prior_mean = np.array([param_priors[col]['mean'] for col in scaled_data.columns])
                post_mean = post_cov @ (prior_precision @ prior_mean + X.T @ y)
                
            except np.linalg.LinAlgError:
                print("Warning: Using pseudoinverse calculation")
                post_cov = np.linalg.pinv(prior_precision + X.T @ X + reg_matrix)
                post_mean = post_cov @ (X.T @ y)  # Ignore prior mean when numerically unstable
            
            # Store model parameters
            self.params = {
                'mean': post_mean,
                'cov': post_cov,
                'feature_names': scaled_data.columns,
                'global_scale': global_scale,
                'group_scales': group_scales,
                'param_priors': param_priors,
                'time_decay': time_weights
            }
            
            return True
            
        except Exception as e:
            print(f"BVAR model training error: {str(e)}")
            return False

def predict(self, prices, features):
        """Use model for prediction"""
        try:
            if self.params is None:
                raise Exception("Model not trained")
            
            predictions = []
            predictions.extend(prices[:1])
            
            for i in range(1, len(prices)):
                try:
                    if i < self.look_back:
                        predictions.append(float(prices[i-1]))
                        continue
                        
                    window_data = self.prepare_data(prices, features, i, is_training=False)
                    if window_data.empty:
                        predictions.append(float(prices[i-1]))
                        continue
                    
                    scaled_data = self.scaler.transform(window_data)
                    last_data = scaled_data[-1].reshape(1, -1)
                    
                    # Basic prediction
                    pred_mean = np.dot(last_data, self.params['mean'])
                    pred_var = np.dot(np.dot(last_data, self.params['cov']), last_data.T)
                    
                    # Calculate time-dependent uncertainty
                    horizon = max(0, i - self.look_back)
                    base_uncertainty = 0.015
                    growth_rate = 0.012
                    time_uncertainty = 1.0 + base_uncertainty * (1 - np.exp(-growth_rate * horizon))
                    
                    # Improved volatility calculation - add safety checks
                    def safe_volatility(prices_window):
                        if len(prices_window) < 2:  # If too few data points
                            return 0.0
                        mean_price = np.mean(prices_window)
                        if mean_price == 0:  # Avoid division by zero
                            return 0.0
                        return np.std(prices_window) / mean_price
                    
                    # Use safe volatility calculation
                    day1_vol = safe_volatility(prices[max(0, i-1):i])
                    ultra_short_vol = safe_volatility(prices[max(0, i-3):i])
                    short_vol = safe_volatility(prices[max(0, i-7):i])
                    long_vol = safe_volatility(prices[max(0, i-20):i]
                    
                    # Ensure all volatility rates are valid numbers
                    day1_vol = 0.0 if np.isnan(day1_vol) else day1_vol
                    ultra_short_vol = 0.0 if np.isnan(ultra_short_vol) else ultra_short_vol
                    short_vol = 0.0 if np.isnan(short_vol) else short_vol
                    long_vol = 0.0 if np.isnan(long_vol) else long_vol
                    
                    volatility_factor = np.clip(
                        0.3 * day1_vol +
                        0.3 * ultra_short_vol + 
                        0.25 * short_vol + 
                        0.15 * long_vol,
                        0.0015,  # Minimum value
                        0.03    # Maximum value
                    )
                    
                    pred_std = np.sqrt(pred_var).item() * time_uncertainty * (1 + volatility_factor)
                    
                    # Safe sample count calculation
                    base_samples = 1500
                    extra_samples = int(np.clip(400 * volatility_factor / 0.03, 0, 500))  # Limit extra sampling range
                    n_samples = base_samples + extra_samples
                    
                    # Multiple sampling
                    all_samples = []
                    for _ in range(4):
                        samples = norm.rvs(loc=pred_mean, scale=pred_std, size=n_samples)
                        all_samples.extend(samples)
                    
                    pred_samples = np.array(all_samples)
                    
                    # Improved outlier handling
                    q1, q2, q3 = np.percentile(pred_samples, [25, 50, 75])
                    iqr = q3 - q1
                    iqr_factor = np.clip(1.05 + volatility_factor * 2, 1.05, 1.25)
                    lower_bound = q1 - iqr_factor * iqr
                    upper_bound = q3 + iqr_factor * iqr
                    
                    mask = (pred_samples >= lower_bound) & (pred_samples <= upper_bound)
                    filtered_samples = pred_samples[mask]
                    
                    # Improved weight calculation
                    dist_weights = 1 / (1 + np.abs(filtered_samples - q2))
                    trend_weights = 1 / (1 + np.abs(filtered_samples - predictions[-1]))
                    vol_weights = 1 / (1 + volatility_factor * np.abs(filtered_samples - q2))
                    
                    combined_weights = (
                        0.5 * dist_weights / dist_weights.sum() +
                        0.3 * trend_weights / trend_weights.sum() +
                        0.2 * vol_weights / vol_weights.sum()
                    )
                    
                    # Calculate weighted average
                    pred_scaled = np.average(filtered_samples, weights=combined_weights)
                    
                    # Dynamic smoothing
                    if len(predictions) > 1:
                        prev_pred = predictions[-1]
                        smooth_factor = np.clip(0.12 + volatility_factor, 0.12, 0.22)
                        pred_scaled = (1 - smooth_factor) * pred_scaled + smooth_factor * prev_pred
                    
                    # Convert back to original scale
                    full_vector = np.zeros((1, window_data.shape[1]))
                    full_vector[0, window_data.columns.get_loc('price')] = pred_scaled
                    pred_orig = self.scaler.inverse_transform(full_vector)[0, window_data.columns.get_loc('price')]
                    
                    # Prediction interval restriction
                    prev_price = float(prices[i-1])
                    max_change = min(0.05, max(0.02, volatility_factor))
                    pred_orig = np.clip(
                        pred_orig,
                        prev_price * (1 - max_change),
                        prev_price * (1 + max_change)
                    )
                    
                    predictions.append(float(pred_orig))
                    
                except Exception as e:
                    print(f"Prediction error at time point {i}: {str(e)}")
                    predictions.append(float(prices[i-1]))
            
            return np.array(predictions)
            
        except Exception as e:
            print(f"Prediction process error: {str(e)}")
            raise e

def plot_test_predictions(test_prices, test_pred, look_back, save_path='img/test_predictions_hbvar.png'):
    """Plot test set prediction comparison"""
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Set Chinese font
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # To display Chinese labels correctly
        plt.rcParams['axes.unicode_minus'] = False  # To display negative sign correctly
    except:
        print("Warning: Unable to set Chinese font, using default font")
    
    min_len = min(len(test_prices), len(test_pred))
    test_prices = test_prices[:min_len]
    test_pred = test_pred[:min_len]
    
    x = range(len(test_prices))
    
    plt.plot(x, test_prices, label='Actual', color='blue', linewidth=2)
    plt.plot(x, test_pred, label='Predicted', color='red', linewidth=2, linestyle='--')
    
    plt.title('Hierarchical BVAR Model Test Set Predictions', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10)
    
    os.makedirs('img', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_predictions(predictions, targets):
    """Evaluate prediction results
    Args:
        predictions: Predicted values
        targets: True values
    Returns:
        rmse: Root Mean Square Error
        mse: Mean Square Error
        mae: Mean Absolute Error
        mape: Mean Absolute Percentage Error
        r2: R-squared score
    """
    try:
        predictions = np.array(predictions).ravel()
        targets = np.array(targets).ravel()
        
        # Ensure consistent length
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]
        
        # Handle missing values
        predictions = pd.Series(predictions)
        targets = pd.Series(targets)
        
        predictions = predictions.ffill().bfill().values
        targets = targets.ffill().bfill().values
        
        # Calculate evaluation metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        # Calculate MAPE
        mask = targets != 0
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100 if np.any(mask) else np.nan
        
        # Calculate RMSE and R2
        rmse = np.sqrt(mse)
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return rmse, mse, mae, mape if np.isfinite(mape) else np.nan, r2
        
    except Exception as e:
        print(f"Evaluation process error: {str(e)}")
        return float('inf'), float('inf'), float('inf'), float('inf'), float('-inf')

def main():
    # Load data
    day_data = pd.read_csv('data/day.csv')
    month_data = pd.read_csv('data/month.csv')
    
    # Process date format
    day_data['Date'] = pd.to_datetime(day_data['Date'])
    month_data['Date'] = pd.to_datetime(month_data['Date'].astype(str), 
                                      format='%Y%m')
    
    # Set date index
    month_data.set_index('Date', inplace=True)
    day_data.set_index('Date', inplace=True)
    
    # Select monthly indicators
    feature_cols = ['Timber Price Index', 'Chemical Raw Materials Price Index', 
                   'Energy Price Index', 'NHPI']
    month_data = month_data[feature_cols]  # Directly use original monthly data
    
    # Resample monthly data to daily data
    features = month_data.resample('D').ffill()
    features = features.reindex(day_data.index)
    
    # Get price data
    prices = day_data['Price'].values
    
    # Data splitting
    train_size = int(0.8 * len(prices))
    val_size = int(0.9 * len(prices))
    
    train_prices = prices[:train_size]
    val_prices = prices[train_size:val_size]
    test_prices = prices[val_size:]
    
    train_features = features[:train_size]
    val_features = features[train_size:val_size]
    test_features = features[val_size:]
    
    # Train and evaluate model
    model = HierarchicalBVAR(look_back=20)
    if model.fit(train_prices, train_features):
        # Make predictions
        train_pred = model.predict(train_prices, train_features)
        val_pred = model.predict(val_prices, val_features)
        test_pred = model.predict(test_prices, test_features)
        
        # Evaluate results
        train_metrics = evaluate_predictions(train_pred, train_prices[1:])
        val_metrics = evaluate_predictions(val_pred, val_prices[1:])
        test_metrics = evaluate_predictions(test_pred, test_prices[1:])
        
        # Print evaluation results
        print("\nHierarchical BVAR Model Training Set Evaluation Results:")
        print(f"Train RMSE: {train_metrics[0]:.4f}")
        print(f"Train MSE: {train_metrics[1]:.4f}")
        print(f"Train MAE: {train_metrics[2]:.4f}")
        print(f"Train MAPE: {train_metrics[3]:.4f}")
        print(f"Train R2: {train_metrics[4]:.4f}")
        
        print("\nHierarchical BVAR Model Validation Set Evaluation Results:")
        print(f"Val RMSE: {val_metrics[0]:.4f}")
        print(f"Val MSE: {val_metrics[1]:.4f}")
        print(f"Val MAE: {val_metrics[2]:.4f}")
        print(f"Val MAPE: {val_metrics[3]:.4f}")
        print(f"Val R2: {val_metrics[4]:.4f}")
        
        print("\nHierarchical BVAR Model Test Set Evaluation Results:")
        print(f"Test RMSE: {test_metrics[0]:.4f}")
        print(f"Test MSE: {test_metrics[1]:.4f}")
        print(f"Test MAE: {test_metrics[2]:.4f}")
        print(f"Test MAPE: {test_metrics[3]:.4f}")
        print(f"Test R2: {test_metrics[4]:.4f}")
        
        # Use local plotting function
        plot_test_predictions(test_prices, test_pred, model.look_back)

if __name__ == "__main__":
    main()