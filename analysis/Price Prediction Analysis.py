import pandas as pd
import numpy as np

# Read data
df = pd.read_csv('data/prediction_results01.csv')

# 1.  Distribution characteristics of the overall price level
#  Basic statistics
price_stats = df['Ground_truth'].describe()
print("\n1. Statistical characteristics of the actual price:")
print(price_stats)

# Price range distribution
bins = [1100, 1200, 1300, 1400, 1500]  #  Set interval boundaries
labels = ['1200以下', '1200-1300', '1300-1400', '1400以上']
df['price_range'] = pd.cut(df['Ground_truth'], bins=bins, labels=labels)
range_dist = df['price_range'].value_counts().sort_index()
print("\n2. Price range distribution:")
print(range_dist)
print("\nPercentage of each range:")
print((range_dist/len(df)*100).round(2), '%')

# 2. Performance of the predicted price
pred_stats = df['Predicted'].describe()
print("\n3. Statistical characteristics of the predicted price:")
print(pred_stats)

# Systematic deviation between the predicted price and the actual price
df['bias'] = df['Predicted'] - df['Ground_truth']
bias_stats = df['bias'].describe()
print("\n4. Statistical characteristics of the prediction bias:")
print(bias_stats)

# 3. Prediction performance in different price ranges
error_by_range = df.groupby('price_range').agg({
    'bias': ['count', 'mean', 'std',
             lambda x: np.mean(np.abs(x)),  # MAE
             lambda x: np.mean(np.abs(x/df.loc[x.index, 'Ground_truth']))*100  # MAPE
    ]
}).round(4)
error_by_range.columns = ['Sample size', 'Average bias', 'Standard deviation', 'MAE', 'MAPE(%)']
print("\n5.  Prediction performance in different price ranges:")
print(error_by_range)