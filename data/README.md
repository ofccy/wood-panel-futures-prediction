# Data Description

This document provides detailed information about the datasets used in the wood-based panel futures price prediction research.

## Overview

The research utilizes two main datasets:
1. Daily wood-based panel futures price data (`day.csv`)
2. Monthly supply chain feature data (`month.csv`)

These datasets cover the period from January 2014 to June 2024.

## Daily Price Data (`day.csv`)

### Source
- **Origin**: Dalian Commodity Exchange, China
- **Collection Method**: Sina Finance API
- **Contract Type**: Fiberboard continuous contracts
- **Time Period**: January 2014 to June 2024

### Data Format
The data is organized in two columns:
- **Date**: Trading date in YYYY/M/D format (e.g., 2014/1/2)
- **Price**: Wood-based panel futures price in Yuan (RMB)

Sample data:
```
Date       Price
2014/1/2   1601.514
2014/1/3   1604.976
2014/1/6   1540.361
2014/1/7   1542.669
...
```

### Data Processing Notes
- **Unit Standardization**: The Dalian Commodity Exchange adjusted the quotation unit for fiberboard futures from "Yuan (RMB)/piece" to "Yuan (RMB)/m³" on December 3, 2019. Historical prices before this date have been converted assuming equal proportions for three thickness specifications (12mm, 15mm, and 18mm).


## Monthly Supply Chain Data (`month.csv`)

### Source
- **Price Indices**: China's National Bureau of Statistics website
- **NHPI Data**: CSMAR database
- **Time Period**: January 2014 to June 2024

### Data Format
The data includes five columns:
- **Date**: Month in YYYYMM format (e.g., 201401)
- **Timber Price**: Purchase price index for timber and pulp
- **Chemical Price**: Purchase price index for chemical raw materials
- **Energy Price**: Purchase price index for fuel and power
- **NHPI**: National Housing Prosperity Index

Sample data:
```
Date    Timber Price  Chemical Price  Energy Price  NHPI
201401  100           100             100           96.91
201402  100           99.8            99.8          96.91
201403  99.9          99.301          99.5006       96.4
...
```

### Data Processing Notes
- **Index Conversion**: The price indices are set with January 2014 as the base period (100).
- **Frequency**: All data in this file is on a monthly frequency.

## Data Splitting

For model training and evaluation, the data was split as follows:
- **Training Set**: 70% - January 2, 2014, to September 15, 2021
- **Validation Set**: 15% - September 16, 2021, to February 8, 2023
- **Testing Set**: 15% - February 9, 2023, to June 28, 2024

## Terms of Use

This dataset is provided for research purposes only. When using this data, please cite the original paper:

```
[Citation information will be provided after publication]
```
