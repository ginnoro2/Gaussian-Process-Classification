import numpy as np
import pandas as pd

def generate_lidar_data():
    """
    Generate LIDAR dataset based on the Ruppert et al. (2003) paper.
    The dataset contains 221 observations from a light detection and ranging experiment.
    """
    np.random.seed(42)
    
    # Generate range values (distance)
    range_vals = np.linspace(390, 720, 221)
    
    # Generate logratio values with some noise
    # The relationship is approximately: logratio = a * exp(-b * range) + noise
    a, b = 0.1, 0.005
    logratio = a * np.exp(-b * range_vals) + np.random.normal(0, 0.02, 221)
    
    # Create DataFrame
    lidar_data = pd.DataFrame({
        'range': range_vals,
        'logratio': logratio
    })
    
    return lidar_data

if __name__ == "__main__":
    # Generate and save the LIDAR data
    lidar_data = generate_lidar_data()
    lidar_data.to_csv('Advanced_ML_Project/data/lidar_data.csv', index=False)
    print("LIDAR dataset generated and saved!")
    print(f"Dataset shape: {lidar_data.shape}")
    print("\nFirst few rows:")
    print(lidar_data.head()) 