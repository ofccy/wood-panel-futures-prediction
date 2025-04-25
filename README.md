# Wood-based Panel Futures Price Prediction

[![DOI](https://zenodo.org/badge/DOI/10.xxxx/zenodo.xxxxxxx.svg)](https://doi.org/10.xxxx/zenodo.xxxxxxx)

## Overview

This repository contains data and code for the paper: "Wood-based Panel Futures Price Prediction Incorporating Supply Chain Features". The research proposes a novel dual-frequency fusion network (DM-FusionNet) that integrates daily wood-based panel futures prices with monthly supply chain features to improve price prediction accuracy.

### Abstract

This paper proposes a wood-based panel futures price prediction method incorporating supply chain features, aiming to improve prediction accuracy and explore price formation mechanisms. The model constructs a multi-dimensional feature system by integrating upstream material price indices including timber, chemical raw materials, and energy, as well as downstream indicators such as the construction industry prosperity index. To resolve the heterogeneity between daily price data and monthly supply chain features, we design an innovative dual-frequency fusion network (DM-FusionNet). Experimental results demonstrate significant improvements across multiple evaluation metrics, with a 16.8% reduction in MSE and an improved R² value of 0.870 compared to traditional methods.

## Repository Structure

```
├── data/                                                                  # Data files
│   ├── day.csv                                                        # Wood-based panel futures price data (daily)
│   └── month.csv                                                  # Upstream raw material price indices and NHPI (monthly)
├── code/                                                                # Analysis code
│   ├── price_behavior_analysis.py                     # Price behavior analysis
│   └── price_elasticity_analysis.py                     # Price elasticity analysis
├── model/                                                             # Model implementation
│   ├── benchmark_models/                                # Benchmark models
│   │   ├── ann_multi_params.py                         # ANN model
│   │   ├── lstm_multi_params.py                        # LSTM model 
│   │   ├── garch-midas.py                                   # GARCH-MIDAS model
│   │   └── hierarchical_bvar.py                           # Hierarchical_BVAR model
│   └── DM-Fusionnet_model/                            # Proposed model
│       ├── main_DM-FusionNet.py                      # Model implementation
│       ├── main_diagnostics.py                            # Robustness testing
│       └── feature_importance_analysis.py        # Feature importance analysis
├── analysis/                                                           # Results analysis
│   ├── Price_Prediction_Analysis.py                 # Prediction price level analysis
│   ├── temporal_scale_analysis.py                    # Temporal scale prediction analysis  
│   └── market_phase_analysis.py                      # Market phase prediction analysis
└── result/                        # Results
    └── prediction_results01.csv                          # Prediction results
```

## Data Description

The repository includes two main data files:

1. **day.csv**: Daily wood-based panel futures price data from January 1, 2014, to June 30, 2024, collected from the Dalian Commodity Exchange via Sina Finance API. The data  is closing price, with prices standardized to Yuan (RMB)/m³.

2. **month.csv**: Monthly supply chain data including:
   - Timber price index
   - Chemical raw materials price index
   - Energy price index
   - National Housing Prosperity Index (NHPI)

For detailed data descriptions, please refer to the [`data/README.md`](data/README.md) file.

## Model Description

The DM-FusionNet model features two parallel branches:

1. **Daily Data Branch**: A bidirectional LSTM with attention mechanism processes daily futures price data.
2. **Monthly Data Branch**: A lightweight Transformer processes monthly supply chain features.

These branches are combined through a dynamic fusion layer with residual connections.

## Usage

### Prerequisites

```
python>=3.8
numpy
pandas
scikit-learn
torch
matplotlib
seaborn
statsmodels
```

### Training and Prediction

To train the DM-FusionNet model and generate predictions:

```python
# Example usage
python model/DM-Fusionnet_model/main_DM-FusionNet.py
```

### Evaluation

To analyze prediction results across different time scales:

```python
python analysis/temporal_scale_analysis.py
```

To analyze prediction results across different market phases:

```python
python analysis/market_phase_analysis.py
```

## Results

The DM-FusionNet model achieves:
- 16.8% reduction in MSE compared to LSTM
- 24.1% reduction in MAE
- R² value of 0.870
- 96.06% trend prediction accuracy for 60-day predictions

## Citation

If you use this code or data in your research, please cite:

```
[Citation information will be provided after publication]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by the University-Industry Collaborative Education Program [230701456145711].
