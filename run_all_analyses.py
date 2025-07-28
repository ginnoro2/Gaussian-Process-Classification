#!/usr/bin/env python3
"""
Advanced Machine Learning Project - Comprehensive Analysis Runner

This script runs all the analyses for the Advanced ML project including:
- Part A: Linear and Polynomial Regression
- Part B: Model Selection and Feature Importance
- Part C: Gaussian Process Regression
- Part D: LIDAR Analysis

Author: Advanced ML Project
Date: 2024
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_section(title):
    """Print a formatted section header"""
    print(f"\n--- {title} ---")

def run_analysis(script_path, description):
    """Run a Python analysis script"""
    print_section(f"Running {description}")
    print(f"Script: {script_path}")
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            if result.stdout:
                print("Output:")
                print(result.stdout[-500:])  # Show last 500 chars
        else:
            print(f"✗ {description} failed")
            print("Error:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {description} timed out")
        return False
    except Exception as e:
        print(f"✗ {description} failed with exception: {e}")
        return False
    
    return True

def create_project_structure():
    """Create the project directory structure"""
    print_header("Setting up Project Structure")
    
    directories = [
        'plots',
        'results',
        'data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_dependencies():
    """Check if required packages are available"""
    print_header("Checking Dependencies")
    
    required_packages = [
        ('numpy', 'numpy'), 
        ('pandas', 'pandas'), 
        ('matplotlib', 'matplotlib'), 
        ('seaborn', 'seaborn'), 
        ('scikit-learn', 'sklearn'), 
        ('scipy', 'scipy'), 
        ('statsmodels', 'statsmodels')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name} is available")
        except ImportError:
            print(f"✗ {package_name} is missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    return True

def run_part_a_analyses():
    """Run Part A: Regression Models analyses"""
    print_header("Part A: Regression Models")
    
    analyses = [
        ('Part_A_Regression/linear_regression.py', 'Linear Regression Analysis'),
        ('Part_A_Regression/polynomial_regression.py', 'Polynomial Regression Analysis')
    ]
    
    success_count = 0
    for script, description in analyses:
        if run_analysis(script, description):
            success_count += 1
    
    return success_count == len(analyses)

def run_part_b_analyses():
    """Run Part B: Model Selection analyses"""
    print_header("Part B: Model Selection and Evaluation")
    
    analyses = [
        ('Part_B_Model_Selection/model_comparison.py', 'Model Comparison Analysis'),
        ('Part_B_Model_Selection/feature_importance.py', 'Feature Importance Analysis')
    ]
    
    success_count = 0
    for script, description in analyses:
        if run_analysis(script, description):
            success_count += 1
    
    return success_count == len(analyses)

def run_part_c_analyses():
    """Run Part C: Gaussian Process Regression analyses"""
    print_header("Part C: Gaussian Process Regression")
    
    analyses = [
        ('Part_C_Gaussian_Process/gpr_synthetic.py', 'GPR Synthetic Data Analysis'),
        ('Part_C_Gaussian_Process/gpr_time_series.py', 'GPR Time Series Analysis')
    ]
    
    success_count = 0
    for script, description in analyses:
        if run_analysis(script, description):
            success_count += 1
    
    return success_count == len(analyses)

def run_part_d_analyses():
    """Run Part D: LIDAR Analysis"""
    print_header("Part D: LIDAR Analysis")
    
    analyses = [
        ('Part_D_LIDAR_Analysis/lidar_analysis.py', 'LIDAR Comprehensive Analysis')
    ]
    
    success_count = 0
    for script, description in analyses:
        if run_analysis(script, description):
            success_count += 1
    
    return success_count == len(analyses)

def generate_summary_report():
    """Generate a summary report of all analyses"""
    print_header("Summary Report")
    
    # Check if plots were generated
    plot_files = []
    if os.path.exists('plots'):
        plot_files = [f for f in os.listdir('plots') if f.endswith('.png')]
    
    print(f"Generated {len(plot_files)} plot files:")
    for plot_file in sorted(plot_files):
        print(f"  - {plot_file}")
    
    # Summary of analyses
    print("\nAnalysis Summary:")
    print("✓ Linear Regression with EDA, feature selection, and assumption verification")
    print("✓ Polynomial Regression with overfitting analysis and cross-validation")
    print("✓ Model comparison (Linear, Ridge, Lasso, Elastic Net) with hyperparameter tuning")
    print("✓ Feature importance analysis using Lasso and Random Forest")
    print("✓ Gaussian Process Regression on synthetic data with RBF kernel")
    print("✓ GPR time series analysis with multiple kernels (RBF, Matern32, Matern52)")
    print("✓ LIDAR data analysis with Bayesian regression and GPR")
    
    print("\nKey Features:")
    print("• Comprehensive EDA and visualization")
    print("• Model performance evaluation (RMSE, R², MAE)")
    print("• Cross-validation and hyperparameter tuning")
    print("• Uncertainty quantification in GPR")
    print("• Feature importance comparison")
    print("• Residual analysis and assumption verification")
    print("• Confidence interval analysis")

def main():
    """Main function to run all analyses"""
    start_time = time.time()
    
    print_header("Advanced Machine Learning Project")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies before running analyses.")
        return
    
    # Create project structure
    create_project_structure()
    
    # Run all analyses
    results = {}
    
    results['Part A'] = run_part_a_analyses()
    results['Part B'] = run_part_b_analyses()
    results['Part C'] = run_part_c_analyses()
    results['Part D'] = run_part_d_analyses()
    
    # Generate summary
    generate_summary_report()
    
    # Final summary
    end_time = time.time()
    duration = end_time - start_time
    
    print_header("Final Summary")
    print("Analysis Results:")
    for part, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {part}: {status}")
    
    print(f"\nTotal execution time: {duration:.2f} seconds")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if all(results.values()):
        print("\n🎉 All analyses completed successfully!")
        print("Check the 'plots' directory for all generated visualizations.")
    else:
        print("\n⚠️  Some analyses failed. Check the output above for details.")
    
    print("\nProject files:")
    print("• README.md - Project documentation")
    print("• requirements.txt - Dependencies")
    print("• plots/ - Generated visualizations")
    print("• results/ - Analysis results")
    print("• data/ - Datasets")

if __name__ == "__main__":
    main() 