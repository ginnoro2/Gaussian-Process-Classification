#!/usr/bin/env python3
"""
Setup script for Advanced Machine Learning Project

This script helps set up the project environment and install dependencies.
"""

import os
import sys
import subprocess
import platform

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def install_dependencies():
    """Install project dependencies"""
    print_header("Installing Dependencies")
    
    try:
        # Install from requirements.txt
        print("Installing packages from requirements.txt...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print("❌ Failed to install dependencies")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary project directories"""
    print_header("Creating Project Directories")
    
    directories = [
        'plots',
        'results', 
        'data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_imports():
    """Test if all required packages can be imported"""
    print_header("Testing Package Imports")
    
    required_packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'seaborn',
        'sklearn',
        'scipy',
        'statsmodels'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("\n✅ All packages imported successfully")
    return True

def run_quick_test():
    """Run a quick test to verify everything works"""
    print_header("Running Quick Test")
    
    try:
        # Test basic functionality
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Create a simple plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        plt.figure(figsize=(8, 6))
        plt.plot(x, y)
        plt.title('Test Plot')
        plt.xlabel('x')
        plt.ylabel('sin(x)')
        plt.grid(True)
        plt.savefig('plots/test_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Quick test completed successfully")
        print("✅ Test plot saved to plots/test_plot.png")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main setup function"""
    print_header("Advanced ML Project Setup")
    
    print("This script will set up the Advanced Machine Learning project environment.")
    print("It will install dependencies and create necessary directories.")
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        return False
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        print("Please try installing manually: pip install -r requirements.txt")
        return False
    
    # Test imports
    if not test_imports():
        print("\n❌ Setup failed: Package import test failed")
        return False
    
    # Run quick test
    if not run_quick_test():
        print("\n❌ Setup failed: Quick test failed")
        return False
    
    print_header("Setup Complete!")
    print("✅ Project environment is ready")
    print("\nNext steps:")
    print("1. Run the main analysis: python run_all_analyses.py")
    print("2. Or run individual parts:")
    print("   - Part A: python Part_A_Regression/linear_regression.py")
    print("   - Part B: python Part_B_Model_Selection/model_comparison.py")
    print("   - Part C: python Part_C_Gaussian_Process/gpr_synthetic.py")
    print("   - Part D: python Part_D_LIDAR_Analysis/lidar_analysis.py")
    print("\nCheck the README.md file for detailed documentation.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 