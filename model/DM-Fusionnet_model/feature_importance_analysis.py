import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei']  # Normally display Chinese labels
plt.rcParams['axes.unicode_minus'] = False  # Normally display negative sign

# Import existing model and dataset classes
frommain_DM-FusionNet import PriceDataset, FusionModel, TimeConfig, collate_fn
from calculate_perturbation_range import calculate_perturbation_ranges, plot_feature_distributions

def calculate_feature_importance(model, test_dataset, feature_names):
    """
    Calculate importance scores for monthly frequency features
    
    Parameters:
    - model: Trained model
    - test_dataset: Test dataset
    - feature_names: List of feature names
    
    Returns:
    - importance_scores: Feature importance score dictionary
    """
    model.eval()
    
    # Create result storage dictionary
    results = {
        'feature_name': [],
        'importance_score': [],
        'delta_p_plus': [],
        'delta_p_minus': []
    }
    
    # Fixed perturbation percentage
    perturbation_pct = 0.1  # 10% perturbation
    
    with torch.no_grad():
        for feature_idx, feature_name in enumerate(feature_names):
            print(f"Analyzing feature: {feature_name}")
            
            # Get original prediction
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=collate_fn)
            batch = next(iter(test_loader))
            if batch is None:
                continue
                
            day_input, month_input, target = batch
            
            original_output, _, _, _ = model(day_input, month_input)
            original_price = original_output.numpy()
            
            # Positive perturbation
            perturbed_month_input = month_input.clone()
            perturbed_month_input[:, :, feature_idx] *= (1 + perturbation_pct)
            output_plus, _, _, _ = model(day_input, perturbed_month_input)
            price_plus = output_plus.numpy()
            
            delta_p_plus = np.abs(price_plus - original_price) / np.abs(original_price)
            
            # Negative perturbation
            perturbed_month_input = month_input.clone()
            perturbed_month_input[:, :, feature_idx] *= (1 - perturbation_pct)
            output_minus, _, _, _ = model(day_input, perturbed_month_input)
            price_minus = output_minus.numpy()
            
            delta_p_minus = np.abs(price_minus - original_price) / np.abs(original_price)
            
            # Calculate importance score (using average relative change)
            importance_score = (np.mean(delta_p_plus) + np.mean(delta_p_minus)) / 2
            
            # Store results
            results['feature_name'].append(feature_name)
            results['importance_score'].append(importance_score)
            results['delta_p_plus'].append(np.mean(delta_p_plus))
            results['delta_p_minus'].append(np.mean(delta_p_minus))
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('importance_score', ascending=False)
    
    return results_df

def plot_feature_importance(results_df, save_path='img/feature_importance_adaptive.png'):
    """
    Plot feature importance analysis results
    """
    plt.figure(figsize=(12, 6))
    
    # Plot importance score bar graph
    bars = plt.bar(results_df['feature_name'], results_df['importance_score'], color='skyblue', alpha=0.6)
    
    # Set graph format
    plt.title('Monthly Frequency Feature Importance Analysis', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add specific values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Load data
    day_data = pd.read_csv('data/day.csv')
    month_data = pd.read_csv('data/month.csv')
    
    # Feature names
    feature_names = ['Timber Price Index', 
                    'Chemical Raw Materials Price Index',
                    'Energy Price Index', 
                    'NHPI']
    
    # Create time configuration instance
    time_config = TimeConfig()
    
    # Create test dataset
    full_dataset = PriceDataset(day_data, month_data, time_config, scaler=None, is_train=True)
    dataset_size = len(full_dataset)
    test_start_idx = int(0.85 * dataset_size)  # Use last 15% of data as test set
    test_dataset = PriceDataset(
        day_data.iloc[test_start_idx:].reset_index(drop=True),
        month_data,
        time_config,
        scaler=full_dataset.get_scalers(),
        is_train=False
    )
    
    # Load trained model
    model = FusionModel()
    checkpoint = torch.load('models/best_model-0.8703.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Calculate feature importance
    results_df = calculate_feature_importance(model, test_dataset, feature_names)
    
    # Print results
    print("\nFeature Importance Analysis Results:")
    print("=" * 100)
    print(results_df.to_string(index=False, float_format=lambda x: '{:.6f}'.format(x)))
    print("=" * 100)
    
    # Plot feature distributions
    plot_feature_distributions(month_data, feature_names)
    print("\nFeature distribution plot saved to img/feature_distributions.png")
    
    # Plot feature importance visualization results
    plot_feature_importance(results_df)
    print("\nFeature importance analysis chart saved to img/feature_importance_adaptive.png")

if __name__ == "__main__":
    main()