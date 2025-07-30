# Advanced Machine Learning Project

This project covers comprehensive regression analysis including Linear Regression, Polynomial Regression, Model Selection, and Gaussian Process Regression with LIDAR data analysis.

## Project Structure

```
Advanced_ML_Project/
├── Part_A_Regression/          # Linear and Polynomial Regression
├── Part_B_Model_Selection/     # Model comparison and feature importance
├── Part_C_Gaussian_Process/    # GPR on synthetic and real data
├── Part_D_LIDAR_Analysis/     # LIDAR data analysis with Bayesian and GPR
├── data/                      # Datasets
├── results/                   # Results and outputs
├── plots/                     # Generated visualizations
└── requirements.txt           # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the Project

### Part A: Regression Models
```bash
python Part_A_Regression/linear_regression.py
python Part_A_Regression/polynomial_regression.py
```

### Part B: Model Selection
```bash
python Part_B_Model_Selection/model_comparison.py
python Part_B_Model_Selection/feature_importance.py
```

### Part C: Gaussian Process Regression
```bash
python Part_C_Gaussian_Process/gpr_synthetic.py
python Part_C_Gaussian_Process/gpr_time_series.py
```

### Part D: LIDAR Analysis
```bash
python Part_D_LIDAR_Analysis/lidar_analysis.py
```

## Features

1. **Linear Regression**: EDA, feature selection, model fitting, evaluation
2. **Polynomial Regression**: Overfitting analysis, cross-validation
3. **Model Selection**: Linear, Ridge, Lasso, Elastic Net comparison
4. **Feature Importance**: Lasso and Random Forest comparison
5. **Gaussian Process Regression**: Multiple kernels, uncertainty quantification
6. **LIDAR Analysis**: Bayesian regression, GPR with different kernels

## Datasets Used

- Boston Housing Dataset
- Synthetic Student Performance Dataset
- Synthetic datasets for GPR
- Synthetic random LIDAR dataset (221 observations)

## Key Visualizations

- Residual plots
- Model comparison charts
- Feature importance plots
- GPR confidence intervals
- Posterior distributions
- Cross-validation results 
