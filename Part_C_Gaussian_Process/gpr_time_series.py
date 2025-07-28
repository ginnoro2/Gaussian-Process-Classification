import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_time_series_data():
    """Generate synthetic time series data with multiple patterns"""
    print("=== Generating Time Series Data ===")
    
    np.random.seed(42)
    
    # Generate time points
    t = np.linspace(0, 100, 200)
    
    # Generate different time series patterns
    # 1. Trend + Seasonality + Noise
    trend = 0.1 * t
    seasonality = 2 * np.sin(2 * np.pi * t / 20) + 1.5 * np.sin(2 * np.pi * t / 10)
    noise = np.random.normal(0, 0.5, len(t))
    y1 = trend + seasonality + noise
    
    # 2. Temperature-like data (daily temperature simulation)
    t_temp = np.linspace(0, 365, 365)  # One year of daily data
    annual_trend = 15 + 10 * np.sin(2 * np.pi * t_temp / 365)  # Annual cycle
    weekly_variation = 2 * np.sin(2 * np.pi * t_temp / 7)  # Weekly cycle
    noise_temp = np.random.normal(0, 1, len(t_temp))
    y2 = annual_trend + weekly_variation + noise_temp
    
    # 3. Stock price-like data (random walk with trend)
    t_stock = np.linspace(0, 252, 252)  # Trading days in a year
    returns = np.random.normal(0.001, 0.02, len(t_stock))  # Daily returns
    price = 100 * np.exp(np.cumsum(returns))  # Geometric random walk
    y3 = price
    
    print(f"Generated {len(t)} data points for pattern 1")
    print(f"Generated {len(t_temp)} data points for temperature simulation")
    print(f"Generated {len(t_stock)} data points for stock price simulation")
    
    return {
        'pattern1': {'t': t, 'y': y1, 'name': 'Trend + Seasonality'},
        'temperature': {'t': t_temp, 'y': y2, 'name': 'Temperature Data'},
        'stock': {'t': t_stock, 'y': y3, 'name': 'Stock Price Data'}
    }

def fit_gpr_with_different_kernels(X, y, kernel_name='RBF'):
    """Fit GPR with different kernel types"""
    print(f"\n=== Fitting GPR with {kernel_name} Kernel ===")
    
    # Reshape X for sklearn
    X_reshaped = X.reshape(-1, 1)
    
    # Define kernel based on type
    if kernel_name == 'RBF':
        kernel = ConstantKernel(1.0) * RBF(length_scale=5.0)
    elif kernel_name == 'Matern32':
        kernel = ConstantKernel(1.0) * Matern(length_scale=5.0, nu=1.5)
    elif kernel_name == 'Matern52':
        kernel = ConstantKernel(1.0) * Matern(length_scale=5.0, nu=2.5)
    else:
        kernel = ConstantKernel(1.0) * RBF(length_scale=5.0)
    
    # Fit GPR
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=42, alpha=1e-6)
    gpr.fit(X_reshaped, y)
    
    print(f"Fitted kernel: {gpr.kernel_}")
    print(f"Log-marginal-likelihood: {gpr.log_marginal_likelihood():.4f}")
    
    return gpr

def compare_kernels_on_time_series(data_dict):
    """Compare different kernels on time series data"""
    print("\n=== Kernel Comparison on Time Series Data ===")
    
    kernel_types = ['RBF', 'Matern32', 'Matern52']
    results = {}
    
    for data_name, data in data_dict.items():
        print(f"\n--- Analyzing {data['name']} ---")
        results[data_name] = {}
        
        for kernel_type in kernel_types:
            print(f"\nFitting {kernel_type} kernel...")
            
            # Fit GPR
            gpr = fit_gpr_with_different_kernels(data['t'], data['y'], kernel_type)
            
            # Predictions
            X_reshaped = data['t'].reshape(-1, 1)
            y_pred, y_std = gpr.predict(X_reshaped, return_std=True)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(data['y'], y_pred))
            r2 = r2_score(data['y'], y_pred)
            log_likelihood = gpr.log_marginal_likelihood()
            
            results[data_name][kernel_type] = {
                'model': gpr,
                'predictions': y_pred,
                'std': y_std,
                'rmse': rmse,
                'r2': r2,
                'log_likelihood': log_likelihood
            }
            
            print(f"  RMSE: {rmse:.4f}, R²: {r2:.4f}, Log-likelihood: {log_likelihood:.4f}")
    
    return results

def plot_kernel_comparison(data_dict, results):
    """Plot kernel comparison results"""
    print("\n=== Kernel Comparison Visualization ===")
    
    kernel_types = ['RBF', 'Matern32', 'Matern52']
    colors = ['red', 'blue', 'green']
    
    for data_name, data in data_dict.items():
        print(f"\nPlotting results for {data['name']}")
        
        plt.figure(figsize=(20, 12))
        
        # Plot predictions for each kernel
        for i, kernel_type in enumerate(kernel_types):
            plt.subplot(2, 3, i+1)
            
            result = results[data_name][kernel_type]
            t = data['t']
            y = data['y']
            y_pred = result['predictions']
            y_std = result['std']
            
            plt.scatter(t, y, alpha=0.6, color='gray', s=20, label='Data')
            plt.plot(t, y_pred, color=colors[i], linewidth=2, label=f'{kernel_type} prediction')
            plt.fill_between(t, y_pred - 2*y_std, y_pred + 2*y_std, 
                           alpha=0.3, color=colors[i], label='95% confidence interval')
            
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title(f'{data["name"]} - {kernel_type} Kernel')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot uncertainty comparison
        plt.subplot(2, 3, 4)
        for i, kernel_type in enumerate(kernel_types):
            result = results[data_name][kernel_type]
            plt.plot(data['t'], result['std'], color=colors[i], linewidth=2, 
                    label=f'{kernel_type} uncertainty')
        
        plt.xlabel('Time')
        plt.ylabel('Standard Deviation')
        plt.title('Uncertainty Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot performance metrics
        plt.subplot(2, 3, 5)
        metrics = ['RMSE', 'R²', 'Log-likelihood']
        x_pos = np.arange(len(metrics))
        width = 0.25
        
        for i, kernel_type in enumerate(kernel_types):
            result = results[data_name][kernel_type]
            values = [result['rmse'], result['r2'], result['log_likelihood']]
            plt.bar(x_pos + i*width, values, width, label=kernel_type, 
                   alpha=0.7, color=colors[i])
        
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Performance Comparison')
        plt.xticks(x_pos + width, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot residuals
        plt.subplot(2, 3, 6)
        for i, kernel_type in enumerate(kernel_types):
            result = results[data_name][kernel_type]
            residuals = data['y'] - result['predictions']
            plt.scatter(data['t'], residuals, alpha=0.6, color=colors[i], 
                       s=20, label=f'{kernel_type} residuals')
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Residuals')
        plt.title('Residuals Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'plots/kernel_comparison_{data_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

def analyze_confidence_intervals(data_dict, results):
    """Analyze confidence intervals and uncertainty quantification"""
    print("\n=== Confidence Interval Analysis ===")
    
    for data_name, data in data_dict.items():
        print(f"\n--- Analyzing confidence intervals for {data['name']} ---")
        
        plt.figure(figsize=(15, 10))
        
        # Plot confidence intervals for different kernels
        kernel_types = ['RBF', 'Matern32', 'Matern52']
        colors = ['red', 'blue', 'green']
        
        for i, kernel_type in enumerate(kernel_types):
            plt.subplot(2, 3, i+1)
            
            result = results[data_name][kernel_type]
            t = data['t']
            y = data['y']
            y_pred = result['predictions']
            y_std = result['std']
            
            # Plot data and predictions
            plt.scatter(t, y, alpha=0.6, color='gray', s=20, label='Data')
            plt.plot(t, y_pred, color=colors[i], linewidth=2, label='Prediction')
            
            # Plot confidence intervals
            plt.fill_between(t, y_pred - y_std, y_pred + y_std, 
                           alpha=0.3, color=colors[i], label='68% CI')
            plt.fill_between(t, y_pred - 2*y_std, y_pred + 2*y_std, 
                           alpha=0.2, color=colors[i], label='95% CI')
            
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title(f'{kernel_type} - Confidence Intervals')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Uncertainty analysis
        plt.subplot(2, 3, 4)
        for i, kernel_type in enumerate(kernel_types):
            result = results[data_name][kernel_type]
            plt.plot(data['t'], result['std'], color=colors[i], linewidth=2, 
                    label=f'{kernel_type} std')
        
        plt.xlabel('Time')
        plt.ylabel('Standard Deviation')
        plt.title('Uncertainty Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Coverage analysis
        plt.subplot(2, 3, 5)
        coverage_68 = []
        coverage_95 = []
        
        for kernel_type in kernel_types:
            result = results[data_name][kernel_type]
            y_pred = result['predictions']
            y_std = result['std']
            y_true = data['y']
            
            # Calculate coverage
            within_68 = np.sum(np.abs(y_true - y_pred) <= y_std) / len(y_true)
            within_95 = np.sum(np.abs(y_true - y_pred) <= 2*y_std) / len(y_true)
            
            coverage_68.append(within_68)
            coverage_95.append(within_95)
        
        x_pos = np.arange(len(kernel_types))
        width = 0.35
        
        plt.bar(x_pos - width/2, coverage_68, width, label='68% Coverage', alpha=0.7)
        plt.bar(x_pos + width/2, coverage_95, width, label='95% Coverage', alpha=0.7)
        
        plt.xlabel('Kernel Type')
        plt.ylabel('Coverage Rate')
        plt.title('Prediction Interval Coverage')
        plt.xticks(x_pos, kernel_types)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Print coverage statistics
        print(f"Coverage rates for {data['name']}:")
        for i, kernel_type in enumerate(kernel_types):
            print(f"  {kernel_type}: 68% coverage = {coverage_68[i]:.3f}, 95% coverage = {coverage_95[i]:.3f}")
        
        # Uncertainty distribution
        plt.subplot(2, 3, 6)
        for i, kernel_type in enumerate(kernel_types):
            result = results[data_name][kernel_type]
            plt.hist(result['std'], bins=30, alpha=0.6, color=colors[i], 
                    label=f'{kernel_type} std distribution')
        
        plt.xlabel('Standard Deviation')
        plt.ylabel('Frequency')
        plt.title('Uncertainty Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'plots/confidence_intervals_{data_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

def select_best_kernel(data_dict, results):
    """Select the best kernel for each time series"""
    print("\n=== Best Kernel Selection ===")
    
    kernel_types = ['RBF', 'Matern32', 'Matern52']
    
    for data_name, data in data_dict.items():
        print(f"\n--- {data['name']} ---")
        
        # Create comparison table
        comparison_data = []
        for kernel_type in kernel_types:
            result = results[data_name][kernel_type]
            comparison_data.append({
                'Kernel': kernel_type,
                'RMSE': result['rmse'],
                'R²': result['r2'],
                'Log-likelihood': result['log_likelihood']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Select best kernel based on different criteria
        best_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Kernel']
        best_r2 = comparison_df.loc[comparison_df['R²'].idxmax(), 'Kernel']
        best_likelihood = comparison_df.loc[comparison_df['Log-likelihood'].idxmax(), 'Kernel']
        
        print(f"\nBest kernel by RMSE: {best_rmse}")
        print(f"Best kernel by R²: {best_r2}")
        print(f"Best kernel by Log-likelihood: {best_likelihood}")
        
        # Plot kernel comparison
        plt.figure(figsize=(12, 8))
        
        # Performance metrics
        plt.subplot(2, 2, 1)
        metrics = ['RMSE', 'R²', 'Log-likelihood']
        x_pos = np.arange(len(metrics))
        width = 0.25
        
        for i, kernel_type in enumerate(kernel_types):
            result = results[data_name][kernel_type]
            values = [result['rmse'], result['r2'], result['log_likelihood']]
            plt.bar(x_pos + i*width, values, width, label=kernel_type, alpha=0.7)
        
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title(f'Performance Comparison - {data["name"]}')
        plt.xticks(x_pos + width, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Best model visualization
        plt.subplot(2, 2, 2)
        best_kernel = best_r2  # Use R² as primary criterion
        result = results[data_name][best_kernel]
        
        plt.scatter(data['t'], data['y'], alpha=0.6, color='gray', s=20, label='Data')
        plt.plot(data['t'], result['predictions'], color='red', linewidth=2, 
                label=f'Best model ({best_kernel})')
        plt.fill_between(data['t'], result['predictions'] - 2*result['std'], 
                        result['predictions'] + 2*result['std'], 
                        alpha=0.3, color='red', label='95% confidence interval')
        
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Best Model - {data["name"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Kernel characteristics
        plt.subplot(2, 2, 3)
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, kernel_type in enumerate(kernel_types):
            result = results[data_name][kernel_type]
            plt.plot(data['t'], result['std'], color=colors[i], linewidth=2, 
                    label=f'{kernel_type} uncertainty')
        
        plt.xlabel('Time')
        plt.ylabel('Standard Deviation')
        plt.title('Uncertainty Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Summary statistics
        plt.subplot(2, 2, 4)
        summary_stats = []
        for kernel_type in kernel_types:
            result = results[data_name][kernel_type]
            summary_stats.append([
                result['rmse'],
                result['r2'],
                np.mean(result['std']),
                np.std(result['std'])
            ])
        
        summary_df = pd.DataFrame(summary_stats, 
                                columns=['RMSE', 'R²', 'Mean Std', 'Std of Std'],
                                index=kernel_types)
        
        # Create a simple text summary
        plt.text(0.1, 0.9, f"Best Kernel: {best_kernel}", transform=plt.gca().transAxes, 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        plt.text(0.1, 0.7, f"Best R²: {summary_df.loc[best_kernel, 'R²']:.4f}", 
                transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.1, 0.5, f"Best RMSE: {summary_df.loc[best_kernel, 'RMSE']:.4f}", 
                transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.1, 0.3, f"Mean Uncertainty: {summary_df.loc[best_kernel, 'Mean Std']:.4f}", 
                transform=plt.gca().transAxes, fontsize=10)
        
        plt.axis('off')
        plt.title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(f'plots/best_kernel_{data_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run time series GPR analysis"""
    print("Gaussian Process Regression - Time Series Analysis")
    print("=" * 60)
    
    # Generate time series data
    data_dict = generate_time_series_data()
    
    # Compare kernels
    results = compare_kernels_on_time_series(data_dict)
    
    # Plot kernel comparison
    plot_kernel_comparison(data_dict, results)
    
    # Analyze confidence intervals
    analyze_confidence_intervals(data_dict, results)
    
    # Select best kernel
    select_best_kernel(data_dict, results)
    
    print("\n=== Summary ===")
    print("Time series GPR analysis completed!")
    print("All plots saved in the 'plots' directory.")
    print("Analysis complete!")

if __name__ == "__main__":
    main() 