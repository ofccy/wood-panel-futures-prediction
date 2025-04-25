import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class GARCH_MIDAS:
    def __init__(self, look_back=20):
        self.look_back = look_back
        self.garch_model = None
        self.feature_weights = None
        # Default parameters
        self.p = 1
        self.q = 1
        self.momentum_weight = 0.2
        self.trend_weight = 0.2
        self.feature_weight = 0.3
        self.vol_weight = 0.3
        
        # Data processing parameters
        self.max_change = 0.03  # Limit the maximum change in a single prediction
        self.return_clip = 0.05  # Limit return range
        self.price_min = 0.01   # Minimum price limit
        
        self.omega = None  # MIDAS weights
        self.K = 3  # MIDAS lag order
        self.theta = 1.5  # MIDAS parameter
        self.w = 1.2
        
        # New parameters
        self.short_term_weight = 0.7
        self.long_term_weight = 0.3
    
    def set_params(self, **params):
        """Set model parameters"""
        self.p = params.get('p', 1)
        self.q = params.get('q', 1)
        self.momentum_weight = params.get('momentum_weight', 0.2)
        self.trend_weight = params.get('trend_weight', 0.2)
        self.feature_weight = params.get('feature_weight', 0.3)
        self.vol_weight = params.get('vol_weight', 0.3)
        self.max_change = params.get('max_change', 0.03)
        self.return_clip = params.get('return_clip', 0.05)
        self.price_min = params.get('price_min', 0.01)
    
    def prepare_data(self, prices, features, i, is_training=True):
        """Unified data processing function"""
        try:
            # 1. Validate input data
            if i < self.look_back:
                raise ValueError(f"i({i}) must be greater than look_back({self.look_back})")
            if i >= len(prices):
                raise ValueError(f"i({i}) exceeds price sequence length({len(prices)})")
            if i >= len(features):
                raise ValueError(f"i({i}) exceeds feature sequence length({len(features)})")
            
            # 2. Get and process price window
            window = prices[i-self.look_back:i]
            window = pd.Series(window).fillna(method='ffill').fillna(method='bfill').values
            window = np.clip(window, self.price_min, None)  # Avoid zero and negative values
            current_price = window[-1]
            
            # 3. Calculate returns
            returns = []
            for j in range(1, len(window)):
                if window[j-1] > 0:
                    ret = (window[j] - window[j-1]) / window[j-1]
                    ret = np.clip(ret, -self.return_clip, self.return_clip)
                else:
                    ret = 0
                returns.append(ret)
            returns = np.array(returns)
            
            # 4. Calculate momentum
            momentum = (window[-1] - window[0]) / window[0] if window[0] > 0 else 0
            momentum = np.clip(momentum, -self.max_change, self.max_change)
            
            # 5. Calculate trend
            ma_short = np.nanmean(window[-int(self.look_back/2):])
            ma_full = np.nanmean(window)
            trend = (ma_short - ma_full) / ma_full if ma_full > 0 else 0
            trend = np.clip(trend, -self.max_change, self.max_change)
            
            # 6. Calculate feature impact
            feature_impact = 0
            valid_features = 0
            
            for col in features.columns:
                try:
                    curr_val = features.iloc[i][col]
                    prev_val = features.iloc[i-1][col]
                    
                    if prev_val > 0 and not np.isnan(prev_val) and not np.isnan(curr_val):
                        change = (curr_val - prev_val) / prev_val
                        change = np.clip(change, -self.max_change, self.max_change)
                        # Use appropriate weight
                        weight = 0.25 if is_training else self.feature_weights.get(col, 0.25)
                        feature_impact += weight * change
                        valid_features += 1
                except Exception as e:
                    print(f"Feature {col} processing error: {str(e)}")
                    continue
            
            if valid_features > 0:
                feature_impact /= valid_features
            
            # 7. Process invalid values
            returns = pd.Series(returns).fillna(0).values
            momentum = 0 if np.isnan(momentum) else momentum
            trend = 0 if np.isnan(trend) else trend
            feature_impact = 0 if np.isnan(feature_impact) else feature_impact
            
            return {
                'returns': returns,
                'momentum': momentum,
                'trend': trend,
                'feature_impact': feature_impact,
                'current_price': current_price
            }
            
        except Exception as e:
            print(f"Data processing error: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Processing time point: {i}")
            return {
                'returns': np.zeros(self.look_back-1),
                'momentum': 0,
                'trend': 0,
                'feature_impact': 0,
                'current_price': prices[i-1] if i > 0 else self.price_min
            }
def calculate_midas_weights(self):
        """Calculate MIDAS weights"""
        k = np.arange(1, self.K + 1)
        weights = np.exp(self.theta * k - self.w * k**2)
        self.omega = weights / np.sum(weights)
        return self.omega
    
    def calculate_long_run_variance(self, features):
        """Calculate long-term variance component (based on monthly data)"""
        if self.omega is None:
            self.calculate_midas_weights()
            
        tau = np.zeros(len(features))
        
        # Initialize temporary feature weights (if self.feature_weights is None)
        temp_weights = {}
        if self.feature_weights is None:
            for col in features.columns:
                temp_weights[col] = 0.25
        else:
            temp_weights = self.feature_weights
        
        for i in range(len(features)):
            if i < self.K:
                continue
                
            weighted_sum = 0
            valid_count = 0
            
            # Use exponential decay weights
            for k in range(self.K):
                feature_sum = 0
                feature_count = 0
                decay = np.exp(-k)  # Add time decay factor
                
                for col in features.columns:
                    if not np.isnan(features.iloc[i-k][col]):
                        weight = temp_weights[col] * decay  # Combine feature weights and time decay
                        feature_sum += features.iloc[i-k][col] * weight
                        feature_count += 1
                
                if feature_count > 0:
                    weighted_sum += self.omega[k] * (feature_sum / feature_count)
                    valid_count += 1
            
            if valid_count > 0:
                tau[i] = np.exp(weighted_sum / valid_count)
            else:
                tau[i] = 1.0
                
        # Smooth processing
        tau = pd.Series(tau).rolling(window=3, min_periods=1).mean().values
        return tau

    def fit(self, prices, features):
        """Train model"""
        try:
            # Initialize feature weights as uniform weights
            self.feature_weights = {col: 0.25 for col in features.columns}
            
            returns = []
            feature_impacts = []
            
            # Calculate daily returns
            for i in range(self.look_back, len(prices)):
                data = self.prepare_data(prices, features, i, is_training=True)
                returns.append(data['returns'][-1])
            
            returns = np.array(returns)
            
            # Calculate long-term variance component
            tau = self.calculate_long_run_variance(features)
            tau = tau[self.look_back:]
            
            # Combine short-term and long-term volatility components
            combined_variance = returns**2 * tau
            
            # Fit GARCH model
            self.garch_model = arch_model(
                returns, 
                vol='Garch', 
                p=self.p, 
                q=self.q,
                mean='Zero'
            )
            self.garch_results = self.garch_model.fit(disp='off')
            
            # Update feature weights
            volatility = self.garch_results.conditional_volatility
            
            # Use volatility and feature correlation to determine weights
            aligned_features = features.iloc[self.look_back:].reset_index(drop=True)
            for col in features.columns:
                try:
                    feature_values = aligned_features[col].values[:len(volatility)]
                    feature_values = pd.Series(feature_values).fillna(method='ffill').fillna(method='bfill').values
                    
                    correlation = np.corrcoef(volatility, feature_values)[0, 1]
                    weight = abs(correlation) if not np.isnan(correlation) else 0.25
                    self.feature_weights[col] = weight
                except:
                    self.feature_weights[col] = 0.25
            
            # Normalize feature weights
            total_weight = sum(self.feature_weights.values())
            if total_weight > 0:
                for key in self.feature_weights:
                    self.feature_weights[key] /= total_weight
            
            print("\nCurrent feature weights:")
            for name, weight in self.feature_weights.items():
                print(f"{name}: {weight:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Training process error: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            self.feature_weights = {col: 0.25 for col in features.columns}
            return False
    
    def predict(self, prices, features):
        """Predict prices"""
        try:
            predictions = []
            tau = self.calculate_long_run_variance(features)
            
            # Use longer-term moving average as baseline
            ma_short = pd.Series(prices).rolling(window=5).mean()
            ma_long = pd.Series(prices).rolling(window=20).mean()
            
            for i in range(self.look_back, len(prices)):
                data = self.prepare_data(prices, features, i, is_training=False)
                
                try:
                    # Short-term volatility (based on GARCH)
                    garch_forecast = self.garch_model.forecast(
                        data['returns'],
                        reindex=False
                    )
                    short_vol = np.sqrt(garch_forecast.variance.values[-1, :][0])
                    long_vol = np.sqrt(tau[i])
                    
                    # Improved volatility combination method
                    volatility = np.clip(
                        self.short_term_weight * short_vol + 
                        self.long_term_weight * long_vol,
                        0, 0.1
                    )
                    
                    # Use double moving average as baseline
                    base_price = (
                        0.7 * ma_short.iloc[i] + 
                        0.3 * ma_long.iloc[i]
                    )
                    
                    # Calculate relative change
                    relative_change = (
                        self.momentum_weight * data['momentum'] +
                        self.trend_weight * data['trend'] +
                        self.feature_weight * data['feature_impact'] +
                        self.vol_weight * volatility * np.sign(data['momentum'])
                    ) * 0.4  # Further reduce change magnitude
                    
                    relative_change = np.clip(relative_change, -self.max_change, self.max_change)
                    pred = base_price * (1 + relative_change)
                    
                    # Add smoothing
                    if len(predictions) > 0:
                        pred = 0.7 * pred + 0.3 * predictions[-1]
                    
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
                    pred = prices[i-1]
                
                predictions.append(pred)
            
            return np.array(predictions)
            
        except Exception as e:
            print(f"Prediction process error: {str(e)}")
            return np.ones(len(prices) - self.look_back) * np.nanmean(prices)

def evaluate_model(predictions, targets):
    """Evaluate model performance"""
    try:
        # Ensure input is 1D array
        predictions = np.array(predictions).ravel()
        targets = np.array(targets).ravel()
        
        # Handle NaN values
        predictions = pd.Series(predictions).fillna(method='ffill').fillna(method='bfill').values
        targets = pd.Series(targets).fillna(method='ffill').fillna(method='bfill').values
        
        # Calculate evaluation metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        
        # Safely calculate MAPE
        # Avoid division by zero
        mask = targets != 0
        if np.any(mask):
            mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
        else:
            mape = np.nan
        
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
# Ensure all metrics are finite values
        if not np.isfinite(mape):
            mape = np.nan
        
        return rmse, mse, mae, mape, r2
        
    except Exception as e:
        print(f"Evaluation process error: {str(e)}")
        return float('inf'), float('inf'), float('inf'), float('inf'), float('-inf')

def plot_test_predictions(test_prices, test_pred, look_back, save_path='img/test_predictions_grach-mids.png'):
    """Plot test set prediction comparison"""
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Create time index
    x = range(len(test_prices[look_back:]))
    
    # Plot actual and predicted values
    plt.plot(x, test_prices[look_back:], label='Actual Value', color='blue', linewidth=2)
    plt.plot(x, test_pred, label='Predicted Value', color='red', linewidth=2, linestyle='--')
    
    plt.title('GARCH-MIDAS Model Test Set Predictions', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10)
    
    # Ensure img directory exists
    os.makedirs('img', exist_ok=True)
    
    # Save image
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load data
    day_data = pd.read_csv('data/day.csv')
    month_data = pd.read_csv('data/month.csv')
    
    # Process date format
    day_data['Date'] = pd.to_datetime(day_data['Date'])
    # Convert month_data date from YYYYMM format to date format
    month_data['Date'] = pd.to_datetime(month_data['Date'].astype(str), format='%Y%m')
    
    # Extract month information
    day_data['Month'] = day_data['Date'].dt.strftime('%Y%m')
    month_data['Month'] = month_data['Date'].dt.strftime('%Y%m')
    
    # Data preprocessing
    prices = day_data['Price'].values
    
    # Process monthly features
    features = pd.DataFrame()
    
    # Direct matching by month
    merged_data = pd.merge(
        day_data[['Date', 'Month', 'Price']], 
        month_data,
        on='Month',
        how='left'
    )
    
    # Extract features
    for col in ['Timber Price Index', 'Chemical Raw Materials Price Index', 
                'Energy Price Index', 'NHPI']:
        feature_name = col.split()[0]
        features[feature_name] = merged_data[col].values
    
    # Data split
    train_size = int(0.8 * len(prices))
    val_size = int(0.9 * len(prices))
    
    train_prices = prices[:train_size]
    val_prices = prices[train_size:val_size]
    test_prices = prices[val_size:]
    
    train_features = features[:train_size]
    val_features = features[train_size:val_size]
    test_features = features[val_size:]
    
    # Train model
    model = GARCH_MIDAS()
    if model.fit(train_prices, train_features):
        # Predict
        train_pred = model.predict(train_prices, train_features)
        val_pred = model.predict(val_prices, val_features)
        test_pred = model.predict(test_prices, test_features)
        
        # Evaluate
        train_metrics = evaluate_model(train_pred, train_prices[model.look_back:])
        val_metrics = evaluate_model(val_pred, val_prices[model.look_back:])
        test_metrics = evaluate_model(test_pred, test_prices[model.look_back:])
        
        # Print results
        print("\nTraining Set Evaluation Results:")
        print(f"Train RMSE: {train_metrics[0]:.4f}")
        print(f"Train MSE: {train_metrics[1]:.4f}")
        print(f"Train MAE: {train_metrics[2]:.4f}")
        print(f"Train MAPE: {train_metrics[3]:.4f}")
        print(f"Train R2: {train_metrics[4]:.4f}")
        
        print("\nValidation Set Evaluation Results:")
        print(f"Val RMSE: {val_metrics[0]:.4f}")
        print(f"Val MSE: {val_metrics[1]:.4f}")
        print(f"Val MAE: {val_metrics[2]:.4f}")
        print(f"Val MAPE: {val_metrics[3]:.4f}")
        print(f"Val R2: {val_metrics[4]:.4f}")
        
        print("\nTest Set Evaluation Results:")
        print(f"Test RMSE: {test_metrics[0]:.4f}")
        print(f"Test MSE: {test_metrics[1]:.4f}")
        print(f"Test MAE: {test_metrics[2]:.4f}")
        print(f"Test MAPE: {test_metrics[3]:.4f}")
        print(f"Test R2: {test_metrics[4]:.4f}")
        
        # Plot and save test set prediction results
        plot_test_predictions(test_prices, test_pred, model.look_back)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    if not os.path.exists('results'):
        os.makedirs('results')
    main()