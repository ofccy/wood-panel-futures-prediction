import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score,mean_absolute_error
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime
look_back = 20
# Data loading and preprocessing
class DayDataset(Dataset):
    def __init__(self, day_data, month_data):
        self.day_data = day_data.copy()
        self.month_data = month_data.copy()

        # Ensure that the "Date" field in the data is in the datetime format
        self.day_data['Date'] = pd.to_datetime(self.day_data['Date'], errors='coerce')
        self.day_data.dropna(subset=['Date'], inplace=True)

        # Ensure that the monthly-frequency data in the YYYYMM format is converted into datetime and then broadcasted
        self.month_data['Date'] = pd.to_datetime(self.month_data['Date'], format='%Y%m', errors='coerce').dt.to_period(
            'M')
        self.month_data.dropna(subset=['Date'], inplace=True)

        # Merge the daily-frequency data and the monthly-frequency data
        self.day_data['Month'] = self.day_data['Date'].dt.to_period('M')
        self.day_data = self.day_data.merge(self.month_data, left_on='Month', right_on='Date', how='left')
        self.day_data.drop(columns=['Month', 'Date_y'], inplace=True)  # Delete redundant columns
        self.day_data.rename(columns={'Date_x': 'Date'}, inplace=True)

        # self.day_data.to_csv('data.csv')

    def __len__(self):
        return len(self.day_data) - look_back

    def __getitem__(self, idx):

        try:
            if idx + look_back >= len(self.day_data):
                raise IndexError("The dataset index is out of range")

            # Daily frequency features: price sequence of the past 30/look_back days
            day_series = self.day_data['Price'].values[idx:idx + look_back]
            # Monthly frequency features: 4 indices of the month
            month_features = self.day_data.iloc[idx + look_back][
                ['Timber Price Index', 'Chemical Raw Materials Price Index', 'Energy Price Index', 'NHPI']].values
            # Prediction target: the price on the (idx + 30)-th day
            target = self.day_data.iloc[idx + look_back]['Price']

            # Convert to float type, ensure the data is numerical
            day_series = pd.to_numeric(day_series, errors='coerce')
            month_features = pd.to_numeric(month_features, errors='coerce')
            target = pd.to_numeric(target, errors='coerce')

            # Raise an exception if it contains np.nan or invalid values
            if np.any(np.isnan(day_series)) or np.any(np.isnan(month_features)) or np.isnan(target):
                raise ValueError(f"There are NaN values or invalid data, skip the index {idx}")

            # Return "day_series" and the additional features "features"
            return torch.tensor(day_series, dtype=torch.float32), torch.tensor(month_features,
                                                                               dtype=torch.float32), torch.tensor(
                target, dtype=torch.float32)

        except Exception as e:
            warnings.warn(f"Skip the data at index {idx}: {e}")
            return None


# Use the `collate_fn` in the `DataLoader` to skip the `None` data
def collate_fn(batch):
    # When reading batch data, if the batch contains `None` data, we can skip this data
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None  # Handle the case of an empty batch
    else:
        return torch.utils.data.default_collate(batch)



# LSTM Model for price prediction
class LSTMModel(nn.Module):
    def __init__(self, input_dim=look_back + 4, hidden_dim=128, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_dim

        self.input_norm = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(
            input_dim,  # 30 days price + 4 monthly features
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, day_input, month_input):
        # Combine price and monthly features
        x = torch.cat([day_input, month_input], dim=-1)

        # Normalize
        x = self.input_norm(x)
        # LSTM processing
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        # Final prediction
        output = self.fc(lstm_out[:, -1, :])

        return output


# Improved training function
criterion = nn.MSELoss()

def train_model(model, train_loader, val_loader, epochs=1000, patience=150):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_val = float('inf')
    patience_counter = 0
    max_grad_norm = 0.8

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        for batch in train_loader:
            if batch is None:
                continue

            day_input, month_input, target = batch
            optimizer.zero_grad()
            output = model(day_input, month_input)
            mse_loss = criterion(output.squeeze(), target)
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += mse_loss.item()
            batch_count += 1


        # Calculate the average training loss
        avg_train_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        val_loss, r2, rmse, mse, mape, mae = validate_model(model, val_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"RMSE: {rmse:.4f}, "
                  f"MSE: {mse:.4f}, "
                  f"MAPE: {mape:.4f}, "
                  f"MAE: {mae:.4f}, "
                  f"R2: {r2:.4f}")

        if rmse < best_val:
            best_val = rmse
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_lstm_multi_params_model.pth')
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print("Early stopping triggered!")
        #         break

    if best_val != float('inf'):
        model.load_state_dict(torch.load('models/best_lstm_multi_params_model.pth'))

    return model


def validate_model(model, val_loader):
    """Validation function"""
    model.eval()
    total_loss = 0
    batch_count = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            day_input, month_input, target = batch
            output = model(day_input, month_input)
            # Calculate the loss
            mse_loss = criterion(output.squeeze(), target)
            total_loss += mse_loss.item()
            batch_count += 1
            all_preds.extend(output.squeeze().tolist())
            all_targets.extend(target.tolist())

    if batch_count == 0:
        return float('inf'), float('-inf'), float('inf'), float('inf'), float('inf'), float('inf')

    # Convert to numpy array
    preds = np.array(all_preds).reshape(-1, 1)
    targets = np.array(all_targets).reshape(-1, 1)

    # Inverse normalization
    preds = scaler.inverse_transform(preds)
    targets = scaler.inverse_transform(targets)

    # Calculate evaluation metrics
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    mape = mean_absolute_percentage_error(targets, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, preds)

    return total_loss / batch_count, r2, rmse, mse, mape, mae

def plot_predictions(model, test_loader, test_day_data, scaler):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            day_input, month_input, target = batch
            output = model(day_input, month_input)
            all_preds.extend(output.squeeze().tolist())
            all_targets.extend(target.tolist())


    # Convert to numpy array and perform inverse normalization
    preds = np.array(all_preds).reshape(-1, 1)
    targets = np.array(all_targets).reshape(-1, 1)
    preds = scaler.inverse_transform(preds)
    targets = scaler.inverse_transform(targets)

    # Retrieve the corresponding dates
    test_dates = test_day_data['Date'].iloc[look_back:look_back + len(preds)]


    # Create a chart
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, targets, label='Ground-truth', color='blue')
    plt.plot(test_dates, preds, label='Predicted', color='red')

    # Set the chart format
    plt.grid(True)
    plt.title('LSTM(Multi Para) -Test Set Predictions vs Ground Truth')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Set the date format
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    # Set the x-axis date format
    plt.gcf().autofmt_xdate()  # Automatically rotate and format date labels

    # Set the y-axis range to ensure it includes all data points
    plt.ylim(min(np.min(targets), np.min(preds)) * 0.95,
             max(np.max(targets), np.max(preds)) * 1.05)

    # Save the chart
    plt.savefig('img/test_predictions_lstm_multi_params.png')
    plt.close()

    prediction_df = pd.DataFrame({
    'Date': test_dates,
    'Ground_Truth': targets.flatten(),
    'Predicted': preds.flatten()
})
    prediction_df.to_csv('results/test_lstm_multi_params_predictions.csv', index=False)

    # Calculate and return evaluation metrics
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    mape = mean_absolute_percentage_error(targets, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, preds)

    return rmse, mse, mape, mae, r2


# Data loading and splitting
day_data = pd.read_csv('data/day.csv',header=0, names=['Date', 'Price'])  # Add column names
month_data = pd.read_csv('data/month.csv',header=0, names=['Date', 'Timber Price Index', 'Chemical Raw Materials Price Index', 'Energy Price Index', 'NHPI'])  # Add column names


# Ensure that all numerical columns are converted to numerical data types, force conversion and set errors to NaN
day_data['Price'] = pd.to_numeric(day_data['Price'], errors='coerce')
month_data[['Timber Price Index', 'Chemical Raw Materials Price Index', 'Energy Price Index', 'NHPI']] = month_data[['Timber Price Index', 'Chemical Raw Materials Price Index', 'Energy Price Index', 'NHPI']].apply(pd.to_numeric, errors='coerce')

day_data['Date'] = pd.to_datetime(day_data['Date'], errors='coerce')  # Convert the date column
month_data['Date'] = pd.to_datetime(month_data['Date'], format='%Y%m', errors='coerce')  # Convert the date column

# Data loading and normalization
scaler = MinMaxScaler(feature_range=(0, 1))

# Dataset `Price` normalization
day_data['Price'] = scaler.fit_transform(day_data[['Price']])



# Data partitioning
train_day_data = day_data[:int(0.8 * len(day_data))]
val_day_data = day_data[int(0.8 * len(day_data)):int(0.9 * len(day_data))]
test_day_data = day_data[int(0.9 * len(day_data)):]


batch_size = 48

# Generate dataset
train_dataset = DayDataset(train_day_data, month_data)
val_dataset = DayDataset(val_day_data, month_data)
test_dataset = DayDataset(test_day_data, month_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Sequentially sample the validation set to ensure that all data in `val_data` is used in each validation
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), sampler=SequentialSampler(val_dataset), collate_fn=collate_fn)

# Sequentially sample the test set
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), sampler=SequentialSampler(test_dataset), collate_fn=collate_fn)

# for i in range(10):
# Model initialization and training
model = LSTMModel()
# Train the model
# model = train_model(model, train_loader, val_loader)

model.load_state_dict(torch.load('models/lstm10/best_test_lstm_multi_params_model_202502111705_R2_0.8227.pth'))

# Evaluate the model
rmse, mse, mape, mae, r2 = plot_predictions(model, test_loader, test_day_data, scaler)
print(f"Test Results: RMSE: {rmse:.4f}, MSE: {mse:.4f}, MAPE: {mape:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


