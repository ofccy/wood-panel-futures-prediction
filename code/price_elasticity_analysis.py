import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Read data
day_data = pd.read_csv('data/day.csv')
month_data = pd.read_csv('data/month.csv')

# Convert daily frequency data to monthly frequency
day_data['Date'] = pd.to_datetime(day_data['Date'])
day_data['YearMonth'] = day_data['Date'].dt.to_period('M')
monthly_avg_price = day_data.groupby('YearMonth')['Price'].mean().reset_index()
monthly_avg_price['YearMonth'] = monthly_avg_price['YearMonth'].astype(str).str.replace('-', '')

# Modify date format of monthly data
month_data['Date'] = month_data['Date'].astype(str)

# Merge data
merged_data = pd.merge(monthly_avg_price, month_data,
                       left_on='YearMonth',
                       right_on='Date',
                       how='inner')

# Elasticity coefficient calculation function (covariance/variance method)
def calculate_elasticity(x, y):
    # Calculate covariance
    cov_xy = np.cov(x, y)[0, 1]
    # Calculate variance of x
    var_x = np.var(x)

    # Calculate elasticity coefficient
    if var_x != 0:
        elasticity = cov_xy / var_x
    else:
        elasticity = 0

    return elasticity

# Calculate full sample period elasticity coefficients
columns = ['Timber Price Index', 'Chemical Raw Materials Price Index',
           'Energy Price Index', 'NHPI']

full_sample_elasticities = {}
for col in columns:
    full_sample_elasticities[col] = calculate_elasticity(
        merged_data[col],
        merged_data['Price']
    )

# Print full sample period elasticity results
print("Full Sample Elasticity Results:")
for col, elasticity in full_sample_elasticities.items():
    print(f"{col}: {elasticity}")

# Moving window elasticity coefficient analysis
def calculate_moving_window_elasticities(data, window_size=12):
    results = []

    columns = ['Timber Price Index', 'Chemical Raw Materials Price Index',
               'Energy Price Index', 'NHPI']

    for i in range(window_size, len(data)):
        window = data.iloc[i - window_size:i]

        window_elasticities = {}
        for col in columns:
            window_elasticities[col] = calculate_elasticity(
                window[col],
                window['Price']
            )

        window_elasticities['Period'] = data.iloc[i]['YearMonth']
        results.append(window_elasticities)

    return pd.DataFrame(results)

# Define market phases
def define_market_phases(data):
    # Calculate 20-day moving average line
    data['MA20'] = data['Price'].rolling(window=20).mean()

    # Calculate price change rate
    data['Price_Change_Rate'] = data['Price'].pct_change()

    # Define market phases
    def classify_phase(row):
        if row['Price_Change_Rate'] > 0.05:
            return 'Upward'
        elif row['Price_Change_Rate'] < -0.05:
            return 'Downward'
        else:
            return 'Stable'

    data['Market_Phase'] = data.apply(classify_phase, axis=1)

    return data

# Add market phases to merged data
merged_data = define_market_phases(merged_data)

# Calculate marginal effects for different market phases
def calculate_phase_elasticities(data):
    phases = ['Upward', 'Downward', 'Stable']
    columns = ['Timber Price Index', 'Chemical Raw Materials Price Index',
               'Energy Price Index', 'NHPI']

    phase_elasticities = {}

    for phase in phases:
        phase_data = data[data['Market_Phase'] == phase]

        if len(phase_data) > 0:
            phase_elasticities[phase] = {}
            for col in columns:
                phase_elasticities[phase][col] = calculate_elasticity(
                    phase_data[col],
                    phase_data['Price']
                )

    return phase_elasticities

# Calculate moving window elasticity coefficients
moving_elasticities = calculate_moving_window_elasticities(merged_data)
phase_elasticities = calculate_phase_elasticities(merged_data)

# Print marginal effects for different market phases
print("Market Phase Elasticities:")
for phase, elasticities in phase_elasticities.items():
    print(f"\n{phase} Phase:")
    for factor, elasticity in elasticities.items():
        print(f"{factor}: {elasticity}")

# Visualize moving window elasticity coefficients
moving_elasticities['Period'] = pd.to_datetime(moving_elasticities['Period'], format='%Y%m')

plt.figure(figsize=(15, 6))
plt.plot(moving_elasticities['Period'], moving_elasticities['Timber Price Index'], label='Timber Price Index')
plt.plot(moving_elasticities['Period'], moving_elasticities['Chemical Raw Materials Price Index'],
         label='Chemical Raw Materials Price Index')
plt.plot(moving_elasticities['Period'], moving_elasticities['Energy Price Index'], label='Energy Price Index')
plt.plot(moving_elasticities['Period'], moving_elasticities['NHPI'], label='NHPI')

plt.title('Elasticity of Supply Chain Factors')
plt.xlabel('Time')
plt.ylabel('Elasticity')
plt.legend()

plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()