import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_lidar_data():
    """Generate LIDAR dataset with 221 observations"""
    np.random.seed(42)
    
    # Generate range values (distance)
    range_vals = np.linspace(390, 720, 221)
    
    # Generate logratio values with some noise
    a, b = 0.1, 0.005
    logratio = a * np.exp(-b * range_vals) + np.random.normal(0, 0.02, 221)
    
    return pd.DataFrame({
        'range': range_vals,
        'logratio': logratio
    })

def bayesian_regression_analysis(data, n_samples=100):
    """Question 1: Bayesian Regression on randomly selected 100 points"""
    print("=== Question 1: Bayesian Regression Analysis ===")
    
    # Randomly select 100 points
    sample_data = data.sample(n=n_samples, random_state=42)
    X = sample_data['range'].values.reshape(-1, 1)
    y = sample_data['logratio'].values
    
    # Fit linear regression (Bayesian approach with OLS)
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X)
    
    # Calculate residuals for posterior analysis
    residuals = y - y_pred
    sigma_squared = np.var(residuals)
    
    # Bayesian posterior parameters
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X.flatten()])
    XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    posterior_cov = sigma_squared * XtX_inv
    posterior_mean = XtX_inv @ X_with_intercept.T @ y
    
    # Plot scatter plot with fitted model
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X, y, alpha=0.6, color='blue', label='Data points')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Fitted model')
    plt.xlabel('Range')
    plt.ylabel('Log Ratio')
    plt.title('LIDAR Data: Scatter Plot with Fitted Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot histograms of posterior distributions
    plt.subplot(1, 3, 2)
    intercept_samples = np.random.normal(posterior_mean[0], np.sqrt(posterior_cov[0, 0]), 10000)
    plt.hist(intercept_samples, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Intercept')
    plt.ylabel('Frequency')
    plt.title('Posterior Distribution: Intercept')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    slope_samples = np.random.normal(posterior_mean[1], np.sqrt(posterior_cov[1, 1]), 10000)
    plt.hist(slope_samples, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Slope')
    plt.ylabel('Frequency')
    plt.title('Posterior Distribution: Slope')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/lidar_bayesian_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Model Parameters:")
    print(f"Intercept: {posterior_mean[0]:.6f} ± {np.sqrt(posterior_cov[0, 0]):.6f}")
    print(f"Slope: {posterior_mean[1]:.6f} ± {np.sqrt(posterior_cov[1, 1]):.6f}")
    print(f"R² Score: {r2_score(y, y_pred):.4f}")

def gaussian_process_analysis(data, train_ratio=0.8, kernel_type='RBF'):
    """Question 2: Gaussian Process Regression with different train/test splits"""
    print(f"\n=== Question 2: Gaussian Process Regression ({train_ratio*100:.0f}:{(1-train_ratio)*100:.0f} split) ===")
    
    X = data['range'].values.reshape(-1, 1)
    y = data['logratio'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-train_ratio, random_state=42
    )
    
    # Define kernel
    if kernel_type == 'RBF':
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    elif kernel_type == 'Matern32':
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
    else:
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    
    # Fit GPR
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=42, alpha=1e-6)
    gpr.fit(X_train, y_train)
    
    # Predictions
    y_pred_train, std_train = gpr.predict(X_train, return_std=True)
    y_pred_test, std_test = gpr.predict(X_test, return_std=True)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Training data and predictions
    plt.subplot(1, 3, 1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training data')
    plt.plot(X_train, y_pred_train, color='red', linewidth=2, label='GPR prediction')
    plt.fill_between(X_train.flatten(), 
                     y_pred_train - 2*std_train, 
                     y_pred_train + 2*std_train, 
                     alpha=0.3, color='red', label='95% confidence interval')
    plt.xlabel('Range')
    plt.ylabel('Log Ratio')
    plt.title(f'GPR Training ({train_ratio*100:.0f}% data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Test data and predictions
    plt.subplot(1, 3, 2)
    plt.scatter(X_test, y_test, color='green', alpha=0.6, label='Test data')
    plt.plot(X_test, y_pred_test, color='red', linewidth=2, label='GPR prediction')
    plt.fill_between(X_test.flatten(), 
                     y_pred_test - 2*std_test, 
                     y_pred_test + 2*std_test, 
                     alpha=0.3, color='red', label='95% confidence interval')
    plt.xlabel('Range')
    plt.ylabel('Log Ratio')
    plt.title(f'GPR Testing ({(1-train_ratio)*100:.0f}% data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Full dataset with predictions
    plt.subplot(1, 3, 3)
    X_full = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
    y_pred_full, std_full = gpr.predict(X_full, return_std=True)
    
    plt.scatter(X, y, color='gray', alpha=0.4, label='All data')
    plt.plot(X_full, y_pred_full, color='red', linewidth=2, label='GPR prediction')
    plt.fill_between(X_full.flatten(), 
                     y_pred_full - 2*std_full, 
                     y_pred_full + 2*std_full, 
                     alpha=0.3, color='red', label='95% confidence interval')
    plt.xlabel('Range')
    plt.ylabel('Log Ratio')
    plt.title('GPR Full Dataset Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/lidar_gpr_{train_ratio}_{kernel_type}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"Training RMSE: {train_rmse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Kernel: {gpr.kernel_}")
    
    return {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'kernel': str(gpr.kernel_)
    }

def compare_gpr_results(results_80, results_70):
    """Compare GPR results between 80:20 and 70:30 splits"""
    print("\n=== Question 2.3: Comparison of GPR Results ===")
    
    comparison_data = {
        'Metric': ['Train RMSE', 'Test RMSE', 'Train R²', 'Test R²'],
        '80:20 Split': [results_80['train_rmse'], results_80['test_rmse'], 
                        results_80['train_r2'], results_80['test_r2']],
        '70:30 Split': [results_70['train_rmse'], results_70['test_rmse'], 
                        results_70['train_r2'], results_70['test_r2']]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(comparison_data['Metric']))
    width = 0.35
    
    plt.bar(x - width/2, comparison_data['80:20 Split'], width, label='80:20 Split', alpha=0.8)
    plt.bar(x + width/2, comparison_data['70:30 Split'], width, label='70:30 Split', alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('GPR Performance Comparison: 80:20 vs 70:30 Split')
    plt.xticks(x, comparison_data['Metric'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/lidar_gpr_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def synthetic_function_approximation():
    """Question 3: Approximate synthetic functions using GPR"""
    print("\n=== Question 3: Synthetic Function Approximation ===")
    
    # Function 1: f1(x) = sin(x)/x + ε
    print("\n1. Approximating f1(x) = sin(x)/x + ε")
    x1 = np.linspace(-20, 20, 200)
    x1 = x1[x1 != 0]  # Remove x=0 to avoid division by zero
    y1 = np.sin(x1) / x1 + np.random.normal(0, 0.03, len(x1))
    
    # Fit GPR
    kernel1 = ConstantKernel(1.0) * RBF(length_scale=2.0)
    gpr1 = GaussianProcessRegressor(kernel=kernel1, random_state=42, alpha=1e-6)
    gpr1.fit(x1.reshape(-1, 1), y1)
    
    # Predictions
    x1_pred = np.linspace(-20, 20, 1000)
    x1_pred = x1_pred[x1_pred != 0]
    y1_pred, std1_pred = gpr1.predict(x1_pred.reshape(-1, 1), return_std=True)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(x1, y1, alpha=0.6, color='blue', label='Data points')
    plt.plot(x1_pred, y1_pred, color='red', linewidth=2, label='GPR prediction')
    plt.fill_between(x1_pred, y1_pred - 2*std1_pred, y1_pred + 2*std1_pred, 
                     alpha=0.3, color='red', label='95% confidence interval')
    plt.xlabel('x')
    plt.ylabel('f1(x)')
    plt.title('GPR Approximation: sin(x)/x + ε')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Function 2: Given data points
    print("\n2. Approximating given data points")
    x2 = np.array([-4, -3, -2, -1, 0, 0.5, 1, 2])
    y2 = np.array([-2, 0, -0.5, 1, 2, 1, 0, -1])
    
    kernel2 = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gpr2 = GaussianProcessRegressor(kernel=kernel2, random_state=42, alpha=1e-6)
    gpr2.fit(x2.reshape(-1, 1), y2)
    
    x2_pred = np.linspace(-4, 2, 200)
    y2_pred, std2_pred = gpr2.predict(x2_pred.reshape(-1, 1), return_std=True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(x2, y2, color='green', s=100, label='Data points')
    plt.plot(x2_pred, y2_pred, color='red', linewidth=2, label='GPR prediction')
    plt.fill_between(x2_pred, y2_pred - 2*std2_pred, y2_pred + 2*std2_pred, 
                     alpha=0.3, color='red', label='95% confidence interval')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('GPR Approximation: Given Data Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/synthetic_function_approximation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Function 3: Complex function
    print("\n3. Approximating complex function")
    x3 = np.array([0.178, 0.388, 0.865, 0.697, 0.569, 0.216, 0.733, 0.0179, 0.936, 0.495])
    x1_vals = 2 * x3 + 0.5
    y3 = np.sin(10 * np.pi * x1_vals) / (2 * x1_vals) + (x1_vals - 1)**4
    
    kernel3 = ConstantKernel(1.0) * RBF(length_scale=0.1)
    gpr3 = GaussianProcessRegressor(kernel=kernel3, random_state=42, alpha=1e-6)
    gpr3.fit(x3.reshape(-1, 1), y3)
    
    x3_pred = np.linspace(0, 1, 200)
    y3_pred, std3_pred = gpr3.predict(x3_pred.reshape(-1, 1), return_std=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x3, y3, color='purple', s=100, label='Data points')
    plt.plot(x3_pred, y3_pred, color='red', linewidth=2, label='GPR prediction')
    plt.fill_between(x3_pred, y3_pred - 2*std3_pred, y3_pred + 2*std3_pred, 
                     alpha=0.3, color='red', label='95% confidence interval')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('GPR Approximation: Complex Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/complex_function_approximation.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all LIDAR analyses"""
    print("LIDAR Data Analysis - Comprehensive Regression Study")
    print("=" * 60)
    
    # Generate LIDAR data
    lidar_data = generate_lidar_data()
    print(f"Generated LIDAR dataset with {len(lidar_data)} observations")
    
    # Question 1: Bayesian Regression
    bayesian_regression_analysis(lidar_data)
    
    # Question 2: Gaussian Process Regression
    # 2.1: 80:20 split
    results_80 = gaussian_process_analysis(lidar_data, train_ratio=0.8, kernel_type='RBF')
    
    # 2.2: 70:30 split
    results_70 = gaussian_process_analysis(lidar_data, train_ratio=0.7, kernel_type='RBF')
    
    # 2.3: Compare results
    compare_gpr_results(results_80, results_70)
    
    # 2.4: Squared Exponential kernel
    print("\n=== Question 2.4: Squared Exponential Kernel ===")
    results_se = gaussian_process_analysis(lidar_data, train_ratio=0.8, kernel_type='RBF')
    
    # 2.5: Matern32 kernel
    print("\n=== Question 2.5: Matern32 Kernel ===")
    results_matern = gaussian_process_analysis(lidar_data, train_ratio=0.8, kernel_type='Matern32')
    
    # Question 3: Synthetic function approximation
    synthetic_function_approximation()
    
    print("\nAnalysis complete! All plots saved in the 'plots' directory.")

if __name__ == "__main__":
    main() 