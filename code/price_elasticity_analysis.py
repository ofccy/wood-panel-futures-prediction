import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')



# 读取数据
day_data = pd.read_csv('data/day.csv')
month_data = pd.read_csv('data/month.csv')

# 日频数据转月频
day_data['Date'] = pd.to_datetime(day_data['Date'])
day_data['YearMonth'] = day_data['Date'].dt.to_period('M')
monthly_avg_price = day_data.groupby('YearMonth')['Price'].mean().reset_index()
monthly_avg_price['YearMonth'] = monthly_avg_price['YearMonth'].astype(str).str.replace('-', '')

# 修改月度数据的日期格式
month_data['Date'] = month_data['Date'].astype(str)

# 合并数据
merged_data = pd.merge(monthly_avg_price, month_data,
                       left_on='YearMonth',
                       right_on='Date',
                       how='inner')


# 弹性系数计算函数（协方差/方差法）
def calculate_elasticity(x, y):
    # 计算协方差
    cov_xy = np.cov(x, y)[0, 1]
    # 计算x的方差
    var_x = np.var(x)

    # 计算弹性系数
    if var_x != 0:
        elasticity = cov_xy / var_x
    else:
        elasticity = 0

    return elasticity


# 计算全样本期弹性系数
columns = ['Timber Price Index', 'Chemical Raw Materials Price Index',
           'Energy Price Index', 'NHPI']

full_sample_elasticities = {}
for col in columns:
    full_sample_elasticities[col] = calculate_elasticity(
        merged_data[col],
        merged_data['Price']
    )

# 打印全样本期弹性系数
print("Full Sample Elasticity Results:")
for col, elasticity in full_sample_elasticities.items():
    print(f"{col}: {elasticity}")


# 移动窗口弹性系数分析
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


# 定义市场阶段
def define_market_phases(data):
    # 计算20日移动平均线
    data['MA20'] = data['Price'].rolling(window=20).mean()

    # 计算价格变化率
    data['Price_Change_Rate'] = data['Price'].pct_change()

    # 定义市场阶段
    def classify_phase(row):
        if row['Price_Change_Rate'] > 0.05:
            return 'Upward'
        elif row['Price_Change_Rate'] < -0.05:
            return 'Downward'
        else:
            return 'Stable'

    data['Market_Phase'] = data.apply(classify_phase, axis=1)

    return data


# 在合并数据上添加市场阶段
merged_data = define_market_phases(merged_data)


# 计算不同市场阶段的边际效应
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


# 计算移动窗口弹性系数
moving_elasticities = calculate_moving_window_elasticities(merged_data)
phase_elasticities = calculate_phase_elasticities(merged_data)

# 打印不同市场阶段的边际效应
print("Market Phase Elasticities:")
for phase, elasticities in phase_elasticities.items():
    print(f"\n{phase} Phase:")
    for factor, elasticity in elasticities.items():
        print(f"{factor}: {elasticity}")

# 可视化移动窗口弹性系数
moving_elasticities['Period'] = pd.to_datetime(moving_elasticities['Period'], format='%Y%m')

plt.figure(figsize=(9, 5))
plt.plot(moving_elasticities['Period'], moving_elasticities['Timber Price Index'], label='Timber Price Index')
plt.plot(moving_elasticities['Period'], moving_elasticities['Chemical Raw Materials Price Index'],
         label='Chemical Raw Materials Price Index')
plt.plot(moving_elasticities['Period'], moving_elasticities['Energy Price Index'], label='Energy Price Index')
plt.plot(moving_elasticities['Period'], moving_elasticities['NHPI'], label='NHPI')

plt.title('Dynamic Evolution of Supply Chain Factors\' Marginal Elasticity', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Elasticity Coefficient', fontsize=12)  # 添加了单位描述
plt.legend(fontsize=10)

plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# 添加一条水平参考线表示零值
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# 增加网格便于阅读
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('supply_chain_elasticity.png', dpi=300, bbox_inches='tight')
plt.show()