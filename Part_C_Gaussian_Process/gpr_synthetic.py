import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_synthetic_data():
    """Generate synthetic data using y = sin(x)/x + ε"""
    print("=== Generating Synthetic Data ===")
    
    np.random.seed(42)
    
    # Generate x values
    x = np.linspace(-20, 20, 200)
    x = x[x != 0]  # Remove x=0 to avoid division by zero
    
    # Generate y values with noise
    y_true = np.sin(x) / x
    noise = np.random.normal(0, 0.03, len(x))
    y = y_true + noise
    
    print(f"Generated {len(x)} data points")
    print(f"X range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Y range: [{y.min():.2f}, {y.max():.2f}]")
    
    return x, y, y_true

def fit_gpr_model(X, y):
    """Fit Gaussian Process Regression with RBF kernel"""
    print("\n=== Fitting Gaussian Process Regression ===")
    
    # Reshape X for sklearn
    X_reshaped = X.reshape(-1, 1)
    
    # Define RBF kernel
    kernel = ConstantKernel(1.0) * RBF(length_scale=2.0)
    
    # Fit GPR
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=42, alpha=1e-6)
    gpr.fit(X_reshaped, y)
    
    print(f"Fitted kernel: {gpr.kernel_}")
    print(f"Log-marginal-likelihood: {gpr.log_marginal_likelihood():.4f}")
    
    return gpr

def fit_polynomial_regression(X, y, degree=3):
    """Fit polynomial regression for comparison"""
    print(f"\n=== Fitting Polynomial Regression (Degree {degree}) ===")
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Create pipeline
    poly_model = Pipeline([
        ('poly', poly_features),
        ('regressor', LinearRegression())
    ])
    
    # Fit model
    X_reshaped = X.reshape(-1, 1)
    poly_model.fit(X_reshaped, y)
    
    return poly_model

def plot_gpr_results(X, y, y_true, gpr, poly_model, degree=3):
    """Plot GPR results with uncertainty and comparison"""
    print("\n=== Plotting GPR Results ===")
    
    # Create fine grid for predictions
    X_plot = np.linspace(X.min(), X.max(), 1000)
    X_plot_reshaped = X_plot.reshape(-1, 1)
    
    # Generate true function values for plotting
    y_true_plot = np.sin(X_plot) / X_plot
    
    # GPR predictions
    y_gpr_pred, y_gpr_std = gpr.predict(X_plot_reshaped, return_std=True)
    
    # Polynomial predictions
    y_poly_pred = poly_model.predict(X_plot_reshaped)
    
    # Calculate metrics
    y_gpr_train_pred = gpr.predict(X.reshape(-1, 1))
    y_poly_train_pred = poly_model.predict(X.reshape(-1, 1))
    
    gpr_rmse = np.sqrt(mean_squared_error(y, y_gpr_train_pred))
    poly_rmse = np.sqrt(mean_squared_error(y, y_poly_train_pred))
    gpr_r2 = r2_score(y, y_gpr_train_pred)
    poly_r2 = r2_score(y, y_poly_train_pred)
    
    print(f"GPR - RMSE: {gpr_rmse:.4f}, R²: {gpr_r2:.4f}")
    print(f"Polynomial (degree {degree}) - RMSE: {poly_rmse:.4f}, R²: {poly_r2:.4f}")
    
    # Create comprehensive plot
    plt.figure(figsize=(20, 12))
    
    # Main prediction plot
    plt.subplot(2, 3, 1)
    plt.scatter(X, y, alpha=0.6, color='blue', label='Data points', s=20)
    plt.plot(X_plot, y_true_plot, 'g-', linewidth=2, label='True function', alpha=0.8)
    plt.plot(X_plot, y_gpr_pred, 'r-', linewidth=2, label='GPR prediction')
    plt.fill_between(X_plot, y_gpr_pred - 2*y_gpr_std, y_gpr_pred + 2*y_gpr_std, 
                     alpha=0.3, color='red', label='95% confidence interval')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('GPR vs True Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Polynomial comparison
    plt.subplot(2, 3, 2)
    plt.scatter(X, y, alpha=0.6, color='blue', label='Data points', s=20)
    plt.plot(X_plot, y_true_plot, 'g-', linewidth=2, label='True function', alpha=0.8)
    plt.plot(X_plot, y_poly_pred, 'orange', linewidth=2, label=f'Polynomial (degree {degree})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomial vs True Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals comparison
    plt.subplot(2, 3, 3)
    gpr_residuals = y - y_gpr_train_pred
    poly_residuals = y - y_poly_train_pred
    
    plt.scatter(X, gpr_residuals, alpha=0.6, color='red', label='GPR residuals', s=20)
    plt.scatter(X, poly_residuals, alpha=0.6, color='orange', label=f'Polynomial residuals', s=20)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('Residuals')
    plt.title('Residuals Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Uncertainty analysis
    plt.subplot(2, 3, 4)
    plt.plot(X_plot, y_gpr_std, 'purple', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Standard Deviation')
    plt.title('GPR Uncertainty (Standard Deviation)')
    plt.grid(True, alpha=0.3)
    
    # Performance metrics comparison
    plt.subplot(2, 3, 5)
    metrics = ['RMSE', 'R²']
    gpr_metrics = [gpr_rmse, gpr_r2]
    poly_metrics = [poly_rmse, poly_r2]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = plt.bar(x_pos - width/2, gpr_metrics, width, label='GPR', alpha=0.7, color='red')
    bars2 = plt.bar(x_pos + width/2, poly_metrics, width, label=f'Polynomial (degree {degree})', alpha=0.7, color='orange')
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Performance Comparison')
    plt.xticks(x_pos, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
    
    # Kernel visualization
    plt.subplot(2, 3, 6)
    # Create a simple kernel visualization
    x_kernel = np.linspace(-5, 5, 100)
    kernel_values = []
    
    for x_val in x_kernel:
        # Calculate RBF kernel value
        length_scale = 2.0
        kernel_val = np.exp(-0.5 * (x_val / length_scale)**2)
        kernel_values.append(kernel_val)
    
    plt.plot(x_kernel, kernel_values, 'b-', linewidth=2)
    plt.xlabel('Distance')
    plt.ylabel('Kernel Value')
    plt.title('RBF Kernel Function')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/gpr_synthetic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'gpr_rmse': gpr_rmse,
        'poly_rmse': poly_rmse,
        'gpr_r2': gpr_r2,
        'poly_r2': poly_r2
    }

def analyze_kernel_impact(X, y):
    """Analyze the impact of different kernel parameters"""
    print("\n=== Kernel Impact Analysis ===")
    
    X_reshaped = X.reshape(-1, 1)
    
    # Try different length scales
    length_scales = [0.5, 1.0, 2.0, 5.0, 10.0]
    kernel_results = {}
    
    for ls in length_scales:
        print(f"\nFitting GPR with length_scale = {ls}")
        
        # Define kernel
        kernel = ConstantKernel(1.0) * RBF(length_scale=ls)
        
        # Fit GPR
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=42, alpha=1e-6)
        gpr.fit(X_reshaped, y)
        
        # Predictions
        y_pred = gpr.predict(X_reshaped)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        log_likelihood = gpr.log_marginal_likelihood()
        
        kernel_results[ls] = {
            'rmse': rmse,
            'r2': r2,
            'log_likelihood': log_likelihood,
            'model': gpr
        }
        
        print(f"  RMSE: {rmse:.4f}, R²: {r2:.4f}, Log-likelihood: {log_likelihood:.4f}")
    
    # Plot kernel impact
    plt.figure(figsize=(15, 10))
    
    # Performance metrics
    plt.subplot(2, 3, 1)
    ls_values = list(kernel_results.keys())
    rmse_values = [kernel_results[ls]['rmse'] for ls in ls_values]
    r2_values = [kernel_results[ls]['r2'] for ls in ls_values]
    
    plt.plot(ls_values, rmse_values, 'o-', color='red', linewidth=2, markersize=8, label='RMSE')
    plt.xlabel('Length Scale')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Length Scale')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(ls_values, r2_values, 's-', color='blue', linewidth=2, markersize=8, label='R²')
    plt.xlabel('Length Scale')
    plt.ylabel('R²')
    plt.title('R² vs Length Scale')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # Log-likelihood
    plt.subplot(2, 3, 3)
    log_likelihood_values = [kernel_results[ls]['log_likelihood'] for ls in ls_values]
    plt.plot(ls_values, log_likelihood_values, '^-', color='green', linewidth=2, markersize=8)
    plt.xlabel('Length Scale')
    plt.ylabel('Log Marginal Likelihood')
    plt.title('Log-likelihood vs Length Scale')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # Predictions for different length scales
    X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    
    plt.subplot(2, 3, 4)
    plt.scatter(X, y, alpha=0.6, color='gray', s=20, label='Data')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, ls in enumerate([1.0, 2.0, 5.0]):
        if ls in kernel_results:
            y_pred_plot, _ = kernel_results[ls]['model'].predict(X_plot, return_std=True)
            plt.plot(X_plot, y_pred_plot, color=colors[i], linewidth=2, 
                    label=f'Length scale = {ls}')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predictions for Different Length Scales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Uncertainty comparison
    plt.subplot(2, 3, 5)
    for i, ls in enumerate([1.0, 2.0, 5.0]):
        if ls in kernel_results:
            _, y_std_plot = kernel_results[ls]['model'].predict(X_plot, return_std=True)
            plt.plot(X_plot, y_std_plot, color=colors[i], linewidth=2, 
                    label=f'Length scale = {ls}')
    
    plt.xlabel('x')
    plt.ylabel('Standard Deviation')
    plt.title('Uncertainty for Different Length Scales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Best model performance
    plt.subplot(2, 3, 6)
    best_ls = max(kernel_results.keys(), key=lambda ls: kernel_results[ls]['r2'])
    best_model = kernel_results[best_ls]['model']
    
    y_best_pred, y_best_std = best_model.predict(X_plot, return_std=True)
    
    plt.scatter(X, y, alpha=0.6, color='blue', s=20, label='Data')
    plt.plot(X_plot, y_best_pred, 'r-', linewidth=2, label='Best GPR prediction')
    plt.fill_between(X_plot.flatten(), y_best_pred - 2*y_best_std, y_best_pred + 2*y_best_std, 
                     alpha=0.3, color='red', label='95% confidence interval')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Best Model (Length scale = {best_ls})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/kernel_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return kernel_results

def main():
    """Main function to run GPR analysis on synthetic data"""
    print("Gaussian Process Regression - Synthetic Data Analysis")
    print("=" * 60)
    
    # Generate synthetic data
    X, y, y_true = generate_synthetic_data()
    
    # Fit GPR model
    gpr = fit_gpr_model(X, y)
    
    # Fit polynomial regression for comparison
    poly_model = fit_polynomial_regression(X, y, degree=3)
    
    # Plot results
    metrics = plot_gpr_results(X, y, y_true, gpr, poly_model)
    
    # Analyze kernel impact
    kernel_results = analyze_kernel_impact(X, y)
    
    print("\n=== Summary ===")
    print(f"GPR Performance: RMSE = {metrics['gpr_rmse']:.4f}, R² = {metrics['gpr_r2']:.4f}")
    print(f"Polynomial Performance: RMSE = {metrics['poly_rmse']:.4f}, R² = {metrics['poly_r2']:.4f}")
    print("All plots saved in the 'plots' directory.")
    print("Analysis complete!")

if __name__ == "__main__":
    main() 