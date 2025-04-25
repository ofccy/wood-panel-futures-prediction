import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import os

from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from datetime import datetime

# Configuration Class
class TimeConfig:
    DAY_LOOK_BACK = 20  # Number of days for historical data
    MONTH_LOOK_BACK = 12  # Number of months for historical data

# Training Configuration
class TrainConfig:
    # Dataset split ratio
    TRAIN_RATIO = 0.7  # Training set ratio
    VAL_RATIO = 0.15   # Validation set ratio
    # TEST_RATIO will be automatically 1 - TRAIN_RATIO - VAL_RATIO
    
    BATCH_SIZE = 64  # Increase batch size for stability
    INITIAL_LR = 0.002  # Increase initial learning rate
    # Monthly frequency weight target:
    # 0.3 means 30% monthly, 70% daily
    # 0.5 means 50% monthly, 50% daily
    # 0.7 means 70% monthly, 30% daily
    MONTH_LAMBDA = 0.5
    MIN_EPOCHS = 30  # Increase minimum training epochs
    MAX_EPOCHS = 200  # Increase maximum training epochs
    PATIENCE = 20  # Increase early stopping patience
    WEIGHT_DECAY = 0.005  # Reduce weight decay
    GRADIENT_CLIP = 0.5  # Reduce gradient clipping threshold

# Data Loading and Preprocessing
class PriceDataset(Dataset):
    def __init__(self, day_data, month_data, time_config, scaler=None, is_train=True):
        """Initialize dataset"""
        self.day_data = day_data.copy()
        self.month_data = month_data.copy()
        self.time_config = time_config
        self.is_train = is_train
        
        # Process month frequency data dates
        def convert_month_date(x):
            try:
                if pd.isna(x):
                    return None
                x_str = str(int(x)).zfill(6)
                year = int(x_str[:4])
                month = int(x_str[4:6])
                if year < 1900 or year > 2100 or month < 1 or month > 12:
                    return None
                return pd.to_datetime(f"{year}-{month:02d}-01")
            except:
                return None
        
        # Ensure correct date format
        if not pd.api.types.is_datetime64_any_dtype(self.day_data['Date']):
            self.day_data['Date'] = pd.to_datetime(self.day_data['Date'])
            
        # Process month frequency data dates
        if not pd.api.types.is_datetime64_any_dtype(self.month_data['Date']):
            self.month_data['Date'] = self.month_data['Date'].apply(convert_month_date)
            self.month_data = self.month_data.dropna(subset=['Date'])
            
        # Ensure data is sorted by date
        self.day_data = self.day_data.sort_values('Date')
        self.month_data = self.month_data.sort_values('Date')
        
        # Initialize scaler
        if scaler is None and is_train:
            self.price_scaler = StandardScaler()
            self.feature_scaler = StandardScaler()
            self.fit_scaler()
        else:
            self.price_scaler = scaler['price_scaler']
            self.feature_scaler = scaler['feature_scaler']
            
        if self.price_scaler is not None and self.feature_scaler is not None:
            self.transform_data()
    
    def fit_scaler(self):
        """Fit data standardization"""
        # Prepare daily data
        day_values = self.day_data['Price'].values.reshape(-1, 1)
        self.price_scaler.fit(day_values)
        
        # Prepare monthly data
        month_cols = ['Timber Price Index', 'Chemical Raw Materials Price Index',
                     'Energy Price Index', 'NHPI']
        month_values = self.month_data[month_cols].values
        
        self.feature_scaler.fit(month_values)
    
    def transform_data(self):
        """Standardize data"""
        # Standardize daily data
        self.day_data['Price'] = self.price_scaler.transform(
            self.day_data['Price'].values.reshape(-1, 1)
        ).flatten()
        
        # Standardize monthly data
        month_cols = ['Timber Price Index', 'Chemical Raw Materials Price Index',
                     'Energy Price Index', 'NHPI']
        self.month_data[month_cols] = self.feature_scaler.transform(
            self.month_data[month_cols].values
        )

    def inverse_transform_price(self, price_data):
        """Inverse standardize price data"""
        if isinstance(price_data, torch.Tensor):
            price_data = price_data.detach().cpu().numpy()
        return self.price_scaler.inverse_transform(price_data.reshape(-1, 1))

    def inverse_transform_features(self, feature_data):
        """Inverse standardize feature data"""
        if isinstance(feature_data, torch.Tensor):
            feature_data = feature_data.detach().cpu().numpy()
        return self.feature_scaler.inverse_transform(feature_data)

    def get_scalers(self):
        """Get all scalers"""
        return {
            'price_scaler': self.price_scaler,
            'feature_scaler': self.feature_scaler
        }

    def set_scalers(self, price_scaler, feature_scaler):
        """Set scalers (for validation and test sets)"""
        self.price_scaler = price_scaler
        self.feature_scaler = feature_scaler
        
        # Transform data using set scalers
        self.transform_data()

    def __len__(self):
        return len(self.day_data) - self.time_config.DAY_LOOK_BACK

    def get_previous_months_features(self, current_date):
        """Get feature data for N months before given date"""
        current_month = pd.to_datetime(f"{current_date.year}-{current_date.month:02d}-01")
        month_features = []
        
        # Get dates for previous N months
        for i in range(1, self.time_config.MONTH_LOOK_BACK + 1):
            target_month = current_month - pd.DateOffset(months=i)
            
            # Find corresponding month in monthly data
            month_mask = (
                (self.month_data['Date'].dt.year == target_month.year) &
                (self.month_data['Date'].dt.month == target_month.month)
            )
            month_data = self.month_data[month_mask]
            
            if not month_data.empty:
                # Check if all features have values
                features = month_data.iloc[0][['Timber Price Index', 'Chemical Raw Materials Price Index',
                                             'Energy Price Index', 'NHPI']]
                if features.isna().any():
                    return None
                month_features.append(features.values)
            else:
                return None
        
        return np.array(month_features)  # shape: (month_look_back, 4)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        try:
            # Check if index is valid
            if idx < 0 or idx + self.time_config.DAY_LOOK_BACK >= len(self.day_data):
                return None

            # Get target date (current date)
            target_idx = idx + self.time_config.DAY_LOOK_BACK
            current_date = self.day_data.iloc[target_idx]['Date']
            
            # Get historical price series (previous 20 days)
            history_end_idx = target_idx  # Current date index
            history_start_idx = target_idx - self.time_config.DAY_LOOK_BACK  # 20 days back
            day_series = self.day_data['Price'].values[history_start_idx:history_end_idx]
            
            # Get target price (current date's price)
            target = self.day_data.iloc[target_idx]['Price']

            # Get monthly features (previous 12 months)
            month_features = self.get_previous_months_features(current_date)
            
            # If monthly data is incomplete, return None
            if month_features is None:
                return None

            # Data type conversion and validation
            day_series = np.array(day_series, dtype=np.float32)
            month_features = np.array(month_features, dtype=np.float32)
            target = np.float32(target)

            # Check for invalid values
            if (np.isnan(day_series).any() or 
                np.isnan(month_features).any() or 
                np.isnan(target)):
                return None

            # Convert to tensors (data is already standardized)
            return (torch.tensor(day_series, dtype=torch.float32),
                    torch.tensor(month_features, dtype=torch.float32),
                    torch.tensor(target, dtype=torch.float32))

        except Exception:
            return None


# Collate function to skip None data in DataLoader
def collate_fn(batch):
    # When reading batch data, skip None data
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None  # Handle empty batch
    else:
        return torch.utils.data.default_collate(batch)


# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=1)
        attended_output = torch.sum(x * attention_weights, dim=1)
        return attended_output


# Improved Daily Frequency Branch
class DayBranch(nn.Module):
    def __init__(self, input_dim=TimeConfig.DAY_LOOK_BACK, hidden_dim=128, num_layers=2, dropout=0.3):
        super(DayBranch, self).__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(
            1, hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  # Use bidirectional LSTM
        )
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128)  # Keep output dimension 128
        )

    def forward(self, x):
        # Input normalization
        x = x.unsqueeze(-1)  # [batch_size, seq_len, 1]
        x = self.input_norm(x.squeeze(-1)).unsqueeze(-1)

        # LSTM processing
        lstm_out, _ = self.lstm(x)
        attended = self.attention(lstm_out)

        # Fully connected layer processing
        output = self.fc(attended)

        return output


# Modified MonthBranch Class
class MonthBranch(nn.Module):
    def __init__(self, num_months=TimeConfig.MONTH_LOOK_BACK, num_features=4, hidden_dim=128, dropout=0.3):
        super(MonthBranch, self).__init__()
        
        # Create independent processing channels for each feature
        self.feature_projections = nn.ModuleList([
            nn.Linear(1, hidden_dim // 4) for _ in range(num_features)
        ])
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Feature importance layer
        self.feature_importance = nn.Sequential(
            nn.Linear(hidden_dim, num_features),
            nn.Sigmoid()
        )
        
        # Output projection layer
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x shape: [batch_size, num_months, num_features]
        batch_size = x.size(0)
        
        # Process each feature separately
        feature_outputs = []
        for i in range(4):
            feature = x[:, :, i:i+1]  # [batch_size, num_months, 1]
            projected = self.feature_projections[i](feature)  # [batch_size, num_months, hidden_dim//4]
            feature_outputs.append(projected)
        
        # Combine features
        x = torch.cat(feature_outputs, dim=-1)  # [batch_size, num_months, hidden_dim]
        
        # Transformer processing
        x = self.transformer(x)  # [batch_size, num_months, hidden_dim]
        
        # Get last time step output
        last_hidden = x[:, -1]  # [batch_size, hidden_dim]
        
        # Calculate feature importance
        importance = self.feature_importance(last_hidden)  # [batch_size, num_features]
        
        # Calculate feature mask
        feature_mask = (x.abs().mean(dim=1) > 0).float()  # [batch_size, hidden_dim]
        
        # Output projection
        output = self.output_proj(last_hidden)  # [batch_size, hidden_dim]
        
        # Apply mask
        masked_output = output * feature_mask
        
        return masked_output

    def get_l2_reg(self):
        """Calculate L2 regularization"""
        l2_reg = 0.0
        for name, param in self.named_parameters():
            if 'bias' not in name:  # Only regularize weights
                l2_reg += 0.01 * torch.sum(param ** 2)
        return l2_reg


# Fusion Model with Dynamic Weight Adjustment
class FusionModel(nn.Module):
    def __init__(self, debug_output=False):
        super(FusionModel, self).__init__()
        self.debug_output = debug_output
        self.day_branch = DayBranch()
        self.month_branch = MonthBranch()

        # Keep original weight layer unchanged
        self.weight_layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
            nn.Softplus()
        )

        # Simplified prediction layer, using single linear transformation
        self.fc1 = nn.Linear(128, 1)

        # Improved residual connection
        self.residual = nn.Sequential(
            nn.Linear(TimeConfig.DAY_LOOK_BACK, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Add feature calibration layer
        self.calibration = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, day_input, month_input):
        if self.debug_output:
            print("\n=== Input Check ===")
            print("Monthly input shape:", month_input.shape)
            print("Monthly input is all zero:", torch.all(month_input == 0).item())
            print("Monthly input statistics:", {
                "Mean": month_input.mean().item(),
                "Std": month_input.std().item(),
                "Max": month_input.max().item(),
                "Min": month_input.min().item(),
                "Non-zero element ratio": (month_input != 0).float().mean().item()
            })
        
        # Get features
        day_features = self.day_branch(day_input)
        month_features = self.month_branch(month_input)
        
        if self.debug_output:
            print("\n=== Branch Features ===")
            print("Daily features:", day_features.shape, "\nMean:", day_features.mean().item())
            print("Monthly features:", month_features.shape, "\nMean:", month_features.mean().item())
        
        # Calculate weights
        combined_features = torch.cat([day_features, month_features], dim=1)
        raw_weights = self.weight_layer(combined_features)
        weights = F.softmax(raw_weights, dim=1)
        day_weight = weights[:, 0]
        month_weight = weights[:, 1]
        
        # Directly apply weights
        weighted_features = (day_features * day_weight.unsqueeze(-1) + 
                           month_features * month_weight.unsqueeze(-1))
        
        # Direct linear transformation to output
        main_output = self.fc1(weighted_features)
        residual = self.residual(day_input)
        final_output = main_output + 0.2 * residual
        
        # Get L2 regularization
        l2_reg = self.month_branch.get_l2_reg()
        
        return final_output, l2_reg, day_weight, month_weight


# Modified CustomLoss class
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
    def forward(self, pred, target, model, day_series, month_features):
        # Check for NaN inputs
        if torch.isnan(pred).any() or torch.isnan(target).any():
            print("Warning: Input contains NaN values")
            return torch.tensor(1e6, device=pred.device, dtype=pred.dtype)
        
        # Basic loss: MSE as primary, MAE as secondary
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        
        # Direction consistency loss
        if pred.size(0) > 1:
            diff_pred = pred[1:] - pred[:-1]
            diff_target = target[1:] - target[:-1]
            direction_loss = -torch.mean(torch.sign(diff_pred) * torch.sign(diff_target))
        else:
            direction_loss = torch.tensor(0.0, device=pred.device)
        
        # R² loss calculation
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - pred.squeeze()) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        r2_loss = torch.clamp(1.0 - r2, min=0.0, max=1.0)
        
        # Add weight balance loss
        _, _, day_weight, month_weight = model(day_series, month_features)
        target_month_weight = TrainConfig.MONTH_LAMBDA  # Target monthly weight (0.5)
        weight_balance_loss = 5.0 * torch.mean((month_weight.mean() - target_month_weight) ** 2)
        
        # Total loss
        total_loss = (mse_loss + 
                     0.2 * mae_loss + 
                     0.1 * direction_loss + 
                     0.3 * r2_loss + 
                     weight_balance_loss)  # Add weight balance loss
        
        return total_loss


# Modify train_model function in relevant parts
def train_model(model, train_loader, val_loader, train_dataset):
    print("Starting model training...")
    
    criterion = CustomLoss()
    
    # Restore larger learning rate
    optimizer = torch.optim.AdamW([
        {'params': model.day_branch.parameters(), 'lr': TrainConfig.INITIAL_LR},
        {'params': model.month_branch.parameters(), 'lr': TrainConfig.INITIAL_LR},
        {'params': list(model.weight_layer.parameters()) + 
                  list(model.fc1.parameters()) + 
                  list(model.residual.parameters()), 'lr': TrainConfig.INITIAL_LR}
    ], weight_decay=TrainConfig.WEIGHT_DECAY)
    
    # Use cosine annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TrainConfig.MAX_EPOCHS,
        eta_min=1e-6
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_r2s = []
    val_r2s = []
    
    for epoch in range(TrainConfig.MAX_EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0
        all_train_preds = []
        all_train_targets = []
        
        for batch in train_loader:
            if batch is None:
                continue
            
            day_series, month_features, target = batch
            optimizer.zero_grad()
            
            # Forward propagation
            output, reg_loss, day_weight, month_weight = model(day_series, month_features)
            
            # Check for NaN
            if torch.isnan(output).any():
                print(f"Warning: NaN output in training epoch {epoch+1}")
                continue
            
            # Calculate loss
            pred_loss = criterion(output, target.unsqueeze(-1), model, day_series, month_features)
            # Increase weight balance loss coefficient from 0.1 to 1.0
            loss = pred_loss + reg_loss
            
            # Check if loss is NaN
            if torch.isnan(loss).any():
                print(f"Warning: NaN loss in training epoch {epoch+1}")
                continue
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TrainConfig.GRADIENT_CLIP)
            
            optimizer.step()
            
            # Collect predictions and true values
            pred = train_dataset.inverse_transform_price(output.detach())
            true = train_dataset.inverse_transform_price(target.detach().unsqueeze(-1))
            
            # Check for NaN
            if np.isnan(pred).any() or np.isnan(true).any():
                print(f"Warning: NaN predictions or targets in training epoch {epoch+1}")
                continue
                
            all_train_preds.extend(pred.flatten())
            all_train_targets.extend(true.flatten())
            
            total_loss += loss.item()
            batch_count += 1
        
        # Update learning rate
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                
                day_series, month_features, target = batch
                output, reg_loss, day_weight, month_weight = model(day_series, month_features)
                
                # Check for NaN
                if torch.isnan(output).any():
                    print(f"Warning: NaN output in validation epoch {epoch+1}")
                    continue
                
                pred = train_dataset.inverse_transform_price(output)
                true = train_dataset.inverse_transform_price(target.unsqueeze(-1))
                
                # Check for NaN
                if np.isnan(pred).any() or np.isnan(true).any():
                    print(f"Warning: NaN predictions or targets in validation epoch {epoch+1}")
                    continue
                
                all_val_preds.extend(pred.flatten())
                all_val_targets.extend(true.flatten())
                
                val_loss += criterion(output, target.unsqueeze(-1), model, day_series, month_features).item()
                val_batch_count += 1
        
        # Calculate average loss and R² value
        if batch_count > 0:
            avg_train_loss = total_loss / batch_count
            train_losses.append(avg_train_loss)
            
            # Ensure no NaN values before calculating R²
            if len(all_train_preds) > 0 and len(all_train_targets) > 0:
                # Filter out NaN values
                train_preds = np.array(all_train_preds)
                train_targets = np.array(all_train_targets)
                mask = ~(np.isnan(train_preds) | np.isnan(train_targets))
                if mask.any():
                    train_r2 = r2_score(train_targets[mask], train_preds[mask])
                    train_r2s.append(train_r2)
                else:
                    train_r2 = float('nan')
            else:
                train_r2 = float('nan')
        else:
            avg_train_loss = float('inf')
            train_r2 = float('nan')
        
        if val_batch_count > 0:
            avg_val_loss = val_loss / val_batch_count
            val_losses.append(avg_val_loss)
            
            # Ensure no NaN values before calculating R²
            if len(all_val_preds) > 0 and len(all_val_targets) > 0:
                # Filter out NaN values
                val_preds = np.array(all_val_preds)
                val_targets = np.array(all_val_targets)
                mask = ~(np.isnan(val_preds) | np.isnan(val_targets))
                if mask.any():
                    val_r2 = r2_score(val_targets[mask], val_preds[mask])
                    val_r2s.append(val_r2)
                else:
                    val_r2 = float('nan')
            else:
                val_r2 = float('nan')
        else:
            avg_val_loss = float('inf')
            val_r2 = float('nan')
        
        # Print training information
        print(f'Epoch {epoch+1}:')
        print(f'Training   | Loss: {avg_train_loss:.4f}, R²: {train_r2:.4f}')
        print(f'Validation | Loss: {avg_val_loss:.4f}, R²: {val_r2:.4f}')
        print(f'Learning rates: {[group["lr"] for group in optimizer.param_groups]}')
        
        # Add weight information display
        with torch.no_grad():
            # Get a batch of data for weight display
            sample_batch = next(iter(val_loader))
            if sample_batch is not None:
                day_series, month_features, _ = sample_batch
                _, _, day_weight, month_weight = model(day_series, month_features)
                
                print("\nWeight Distribution:")
                print(f"Daily weight: {day_weight.mean().item():.2%}")
                print(f"Monthly weight: {month_weight.mean().item():.2%}")
        
        print('-' * 70)
        
        # Early stopping check
        if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"Found new best model! Validation Loss = {avg_val_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_r2': train_r2,
                'val_r2': val_r2,
            }, 'models/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= TrainConfig.PATIENCE:
                print(f"Early stopping triggered! Best validation Loss = {best_val_loss:.4f}")
                break
    
    # Load best model
    checkpoint = torch.load('models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_r2s': train_r2s,
        'val_r2s': val_r2s
    }

# In the main program for standardization
def process_data(day_data, month_data):
    """Process daily and monthly data"""
    print("Starting data processing...")
    
    # Convert date format
    day_data['Date'] = pd.to_datetime(day_data['Date'])
    
    # Process monthly data dates
    def convert_to_date(x):
        try:
            # Ensure x is a string of length 6
            x_str= str(x).zfill(6)
            year = int(x_str[:4])
            month = int(x_str[4:6])
            return pd.to_datetime(f"{year}-{month:02d}-01")
        except:
            print(f"Warning: Unable to convert date {x}")
            return None

    month_data['Date'] = month_data['Date'].apply(convert_to_date)
    
    # Handle missing values
    print("Handling missing values...")
    # For monthly data, first forward fill, then backward fill
    month_data = month_data.ffill().bfill()
    
    # Ensure data is sorted by date
    day_data = day_data.sort_values('Date')
    month_data = month_data.sort_values('Date')
    
    print("Data processing completed")
    return day_data, month_data

def evaluate_model(model, data_loader, dataset):
    """Evaluate model performance"""
    model.eval()
    predictions = []
    targets = []
    day_weights = []
    month_weights = []
    
    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
                
            day_series, month_features, target = batch
            output, l2_reg, day_weight, month_weight = model(day_series, month_features)
            
            # Collect weights
            day_weights.extend(day_weight.cpu().numpy())
            month_weights.extend(month_weight.cpu().numpy())
            
            # Inverse standardization
            pred = dataset.inverse_transform_price(output.squeeze())
            true = dataset.inverse_transform_price(target)
            
            predictions.extend(pred.flatten())
            targets.extend(true.flatten())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    mape = mean_absolute_percentage_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    # Calculate average weights
    avg_day_weight = np.mean(day_weights)
    avg_month_weight = np.mean(month_weights)
    
    print(f"Performance Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"\nWeight Distribution:")
    print(f"Daily weight: {avg_day_weight:.2%}")
    print(f"Monthly weight: {avg_month_weight:.2%}")
    
    return {
        'mse': mse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'predictions': predictions,
        'targets': targets,
        'weights': {
            'day': avg_day_weight,
            'month': avg_month_weight
        }
    }

def test_zero_features():
    # Enable debug output in test function
    model = FusionModel(debug_output=True)
    # Load trained model weights
    checkpoint = torch.load('models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create benchmark input data
    day_input = torch.randn(1, 20)              # Normal daily data
    normal_month_input = torch.randn(1, 12, 4)  # Normal monthly data
    
    print("\n=== Benchmark Test (Normal Input) ===")
    with torch.no_grad():
        output_normal, _, _, _ = model(day_input, normal_month_input)
        print("Normal input output:", output_normal.item())
    
    # Test each monthly feature set to zero
    feature_names = ['Timber Price Index', 'Chemical Raw Materials Price Index', 
                     'Energy Price Index', 'NHPI Index']
    for i in range(4):
        # Copy normal input, set i-th column to zero
        test_input = normal_month_input.clone()
        test_input[:, :, i] = 0
        
        print(f"\n=== Test {feature_names[i]} as Zero ===")
        with torch.no_grad():
            output_test, _, _, _ = model(day_input, test_input)
            print(f"{feature_names[i]} as zero output:", output_test.item())
            print(f"Difference from normal output:", abs(output_normal.item() - output_test.item()))
            print(f"Relative difference: {abs(output_normal.item() - output_test.item()) / abs(output_normal.item()):.2%}")

def main():
    """Main function"""
    print("Starting main program...")
    
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
    
    print(f"Dataset Size:\n"
          f"Training set: {len(train_day)} days ({train_day['Date'].min()} to {train_day['Date'].max()})\n"
          f"Validation set: {len(val_day)} days ({val_day['Date'].min()} to {val_day['Date'].max()})\n"
          f"Test set: {len(test_day)} days ({test_day['Date'].min()} to {test_day['Date'].max()})\n"
          f"Total monthly data: {len(month_data)} months")
    
    # Create datasets
    train_dataset = PriceDataset(train_day, month_data, TimeConfig, is_train=True)
    val_dataset = PriceDataset(val_day, month_data, TimeConfig, 
                              scaler=train_dataset.get_scalers(), is_train=False)
    test_dataset = PriceDataset(test_day, month_data, TimeConfig,
                               scaler=train_dataset.get_scalers(), is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=TrainConfig.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=TrainConfig.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=TrainConfig.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = FusionModel(debug_output=False)
    
    # Train model
    # history = train_model(model, train_loader, val_loader, train_dataset)

    # Load pre-trained model
    checkpoint = torch.load('models/best_model-0.8703.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Model evaluation
    print("\nModel Evaluation Results:")
    print("\nTest Set:")
    evaluate_model(model, test_loader, test_dataset)
    
    print("Program completed")

if __name__ == "__main__":
    # test_zero_features()
    main()