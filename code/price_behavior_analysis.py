import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from arch import arch_model
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats
import matplotlib.pyplot as plt


def preprocess_data(daily_file, monthly_file):
    """
    Preprocess daily and monthly data

    Parameters:
    -----------
    daily_file: str, daily data file path
    monthly_file: str, monthly data file path

    Returns:
    --------
    tuple: (daily_df, monthly_df)
    """
    # Read daily data
    daily_df = pd.read_csv('data/day.csv')
    daily_df['Date'] = pd.to_datetime(daily_df['Date'], format='%Y-%m-%d')

    # Ensure price is numeric and handle outliers
    daily_df['Price'] = pd.to_numeric(daily_df['Price'], errors='coerce')
    daily_df['Price'] = daily_df['Price'].replace([np.inf, -np.inf], np.nan)
    daily_df['Price'] = daily_df['Price'].ffill()

    daily_df = daily_df.set_index('Date')
    daily_df = daily_df.sort_index()

    # Read monthly data
    monthly_df = pd.read_csv('data/month.csv')
    monthly_df['Date'] = pd.to_datetime(monthly_df['Date'].astype(str).str.zfill(6), format='%Y%m')

    # Convert all features to numeric
    for col in monthly_df.columns:
        if col != 'Date':
            monthly_df[col] = pd.to_numeric(monthly_df[col], errors='coerce')
            monthly_df[col] = monthly_df[col].replace([np.inf, -np.inf], np.nan)
            monthly_df[col] = monthly_df[col].ffill()

    monthly_df = monthly_df.set_index('Date')
    monthly_df = monthly_df.sort_index()

    return daily_df, monthly_df


def analyze_price_characteristics(daily_df, monthly_df):
    """
    Analyze plywood futures price characteristics and its relationship with the industry chain

    Parameters:
    -----------
    daily_df: DataFrame, daily price data
    monthly_df: DataFrame, monthly industry chain features data

    Returns:
    --------
    dict: dictionary containing various analysis results
    """
    results = {}

    # 1. Trend and Cyclical Analysis
    def analyze_trend_cycle(prices):
        # HP filter decomposition
        cycle, trend = sm.tsa.filters.hpfilter(prices, lamb=10000)

        # Calculate cyclical characteristics
        acf_values = acf(cycle, nlags=30)

        return {
            'trend': trend,
            'cycle': cycle,
            'acf': acf_values
        }

    # 2. Volatility Characteristics Analysis
    def analyze_volatility(prices):
        """
        Analyze price volatility characteristics
        Handle zero and outlier values
        """
        # Replace zero and negative values with previous valid value
        prices_clean = prices.replace(0, np.nan)
        prices_clean = prices_clean.ffill()

        # Calculate returns: use percentage change instead of log returns
        returns = prices_clean.pct_change().dropna()

        # Replace infinite values with column mean if any
        returns = returns.replace([np.inf, -np.inf], np.nan)
        mean_return = returns.mean()
        returns = returns.fillna(mean_return)

        try:
            # Use GARCH(1,1) model to estimate conditional volatility
            model = arch_model(returns, vol='Garch', p=1, q=1, rescale=True)
            results = model.fit(disp='off', show_warning=False)

            # Get conditional volatility
            conditional_vol = results.conditional_volatility

            # Check volatility clustering
            squared_returns = returns ** 2
            vol_clustering = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]

            return {
                'conditional_volatility': conditional_vol,
                'volatility_clustering': vol_clustering,
                'garch_params': results.params
            }
        except Exception as e:
            print(f"GARCH model fitting error: {str(e)}")
            # If GARCH model fails, use simple moving standard deviation
            rolling_std = returns.rolling(window=20).std()
            return {
                'conditional_volatility': rolling_std,
                'volatility_clustering': np.nan,
                'garch_params': None
            }

    # 3. Industry Chain Correlation Analysis
    def analyze_correlation(price_monthly, features_df):
        """
        Analyze correlation between price and industry chain features
        """
        # Calculate lagged correlation coefficients
        max_lag = 6
        lag_corr = {}

        feature_columns = ['Timber Price Index', 'Chemical Raw Materials Price Index',
                           'Energy Price Index', 'NHPI']

        for col in feature_columns:
            corr_lags = []
            for lag in range(max_lag + 1):
                # Ensure clean data
                x = price_monthly[lag:].values
                y = features_df[col][:-lag if lag > 0 else None].values

                # Remove any infinite or NaN values
                mask = np.isfinite(x) & np.isfinite(y)
                x = x[mask]
                y = y[mask]

                if len(x) > 0 and len(y) > 0:
                    try:
                        corr = stats.pearsonr(x, y)[0]
                    except:
                        corr = np.nan
                else:
                    corr = np.nan

                corr_lags.append(corr)
            lag_corr[col] = corr_lags

        # Granger causality test
        granger_results = {}
        for col in feature_columns:
            try:
                data = pd.concat([price_monthly, features_df[col]], axis=1).dropna()
                if len(data) > 0:
                    granger_results[col] = grangercausalitytests(data, maxlag=4, verbose=False)
                else:
                    granger_results[col] = None
            except:
                granger_results[col] = None

        return {
            'lag_correlations': lag_corr,
            'granger_causality': granger_results
        }

    # Execute analysis
    # Process daily data
    daily_price = daily_df['Price']
    results['trend_cycle'] = analyze_trend_cycle(daily_price)
    results['volatility'] = analyze_volatility(daily_price)

    # Convert daily data to monthly for comparison with feature data
    monthly_price = daily_df['Price'].resample('M').last()
    monthly_price.name = 'Plywood_Price'

    # Process monthly data and industry chain features
    results['correlation'] = analyze_correlation(monthly_price, monthly_df)

    return results


def plot_results(results, save_path=None):
    """
    Plot analysis results as separate figures

    Parameters:
    -----------
    results: dict, output from analyze_price_characteristics
    save_path: str, optional, base path for saving images
    """
    # Use matplotlib default style
    plt.style.use('default')


    plt.rcParams.update({'font.size': 12})

    # 1. Trend and Cycle Decomposition
    plt.figure(figsize=(9, 5))
    if 'trend_cycle' in results and results['trend_cycle'] is not None:
        plt.plot(results['trend_cycle']['trend'], label='Long-term Trend Component', color='blue', linewidth=2)
        plt.plot(results['trend_cycle']['cycle'], label='Cyclical and Irregular Component', color='red', alpha=0.6)
        plt.title('Wood-based Panel Futures Price Decomposition (HP Filter, λ=10000)')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (Yuan/m³)', fontsize=12) 
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=10)  
        plt.yticks(fontsize=10)
        if save_path:
            plt.savefig(save_path + '_trend_cycle.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Conditional Volatility
    plt.figure(figsize=(7, 4))
    if 'volatility' in results and results['volatility'] is not None:
        vol_clustering = results['volatility'].get('volatility_clustering', np.nan)
        plt.plot(results['volatility']['conditional_volatility'],
                 color='green', linewidth=2)
        plt.title(f'Conditional Volatility\n(Clustering Coefficient: {vol_clustering:.3f})')
        plt.xlabel('Date')
        plt.ylabel('Volatility (Standard Deviation)', fontsize=12)  
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        if save_path:
            plt.savefig(save_path + '_volatility.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Lagged Correlation Coefficients
    plt.figure(figsize=(7,4))  # Much smaller figure
    if 'correlation' in results and results['correlation'] is not None:
        lag_corr = results['correlation']['lag_correlations']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for (feature, corrs), color in zip(lag_corr.items(), colors):
            # Ensure no infinite values
            corrs = np.array(corrs)
            mask = np.isfinite(corrs)
            if mask.any():
                plt.plot(np.arange(len(corrs))[mask], corrs[mask],
                         label=feature, marker='o', color=color,
                         markersize=5, linewidth=1.5)
        plt.title('Lagged Correlation Coefficients', fontsize=14)
        plt.xlabel('Lag Period (Months)', fontsize=12)
        plt.ylabel('Correlation Coefficient', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 0.85))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path + '_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # Usage example
    daily_df, monthly_df = preprocess_data('data/day.csv', 'data/month.csv')
    results = analyze_price_characteristics(daily_df, monthly_df)
    plot_results(results, save_path='price_analysis_results')