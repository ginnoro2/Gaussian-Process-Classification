import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load California housing dataset and prepare for polynomial regression"""
    print("=== Loading California Housing Dataset for Polynomial Regression ===")
    
    # Load dataset
    housing = fetch_california_housing()
    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    data['target'] = housing.target
    
    # Use only the most correlated feature for polynomial regression demonstration
    # We'll use 'MedInc' (median income) as it has the highest correlation with target
    X = data[['MedInc']].values
    y = data['target'].values
    
    print(f"Using feature: Median Income")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    return X, y

def fit_polynomial_models(X, y, max_degree=5):
    """Fit polynomial regression models of different degrees"""
    print(f"\n=== Fitting Polynomial Models (Degree 1 to {max_degree}) ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = {}
    train_scores = []
    test_scores = []
    train_rmse = []
    test_rmse = []
    
    for degree in range(1, max_degree + 1):
        print(f"\nFitting polynomial regression of degree {degree}")
        
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        
        # Create pipeline with scaling and polynomial features
        model = Pipeline([
            ('poly', poly_features),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse_val = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse_val = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        models[degree] = model
        train_scores.append(train_r2)
        test_scores.append(test_r2)
        train_rmse.append(train_rmse_val)
        test_rmse.append(test_rmse_val)
        
        print(f"Degree {degree}: Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}")
        print(f"Degree {degree}: Train RMSE = {train_rmse_val:.4f}, Test RMSE = {test_rmse_val:.4f}")
    
    return models, train_scores, test_scores, train_rmse, test_rmse, X_train, X_test, y_train, y_test

def plot_overfitting_analysis(train_scores, test_scores, train_rmse, test_rmse):
    """Plot training and test error to demonstrate overfitting"""
    print("\n=== Overfitting Analysis ===")
    
    degrees = range(1, len(train_scores) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # R² scores
    plt.subplot(1, 3, 1)
    plt.plot(degrees, train_scores, 'o-', color='blue', label='Training R²', linewidth=2, markersize=8)
    plt.plot(degrees, test_scores, 's-', color='red', label='Test R²', linewidth=2, markersize=8)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R² Score')
    plt.title('R² Score vs Polynomial Degree')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(degrees)
    
    # RMSE scores
    plt.subplot(1, 3, 2)
    plt.plot(degrees, train_rmse, 'o-', color='blue', label='Training RMSE', linewidth=2, markersize=8)
    plt.plot(degrees, test_rmse, 's-', color='red', label='Test RMSE', linewidth=2, markersize=8)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Polynomial Degree')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(degrees)
    
    # Gap between train and test (overfitting indicator)
    plt.subplot(1, 3, 3)
    r2_gap = np.array(train_scores) - np.array(test_scores)
    rmse_gap = np.array(test_rmse) - np.array(train_rmse)
    
    plt.plot(degrees, r2_gap, 'o-', color='purple', label='R² Gap (Train-Test)', linewidth=2, markersize=8)
    plt.plot(degrees, rmse_gap, 's-', color='orange', label='RMSE Gap (Test-Train)', linewidth=2, markersize=8)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Gap')
    plt.title('Overfitting Gap Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(degrees)
    
    plt.tight_layout()
    plt.savefig('plots/polynomial_overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal degree
    optimal_degree_r2 = degrees[np.argmax(test_scores)]
    optimal_degree_rmse = degrees[np.argmin(test_rmse)]
    
    print(f"\nOptimal degree based on R²: {optimal_degree_r2}")
    print(f"Optimal degree based on RMSE: {optimal_degree_rmse}")
    
    return optimal_degree_r2, optimal_degree_rmse

def cross_validation_analysis(X, y, max_degree=5):
    """Use cross-validation to select the best polynomial degree"""
    print("\n=== Cross-Validation for Model Selection ===")
    
    cv_scores = []
    cv_rmse = []
    
    for degree in range(1, max_degree + 1):
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        
        # Create pipeline
        model = Pipeline([
            ('poly', poly_features),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        # Perform cross-validation
        cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
        cv_rmse_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_rmse_scores = np.sqrt(-cv_rmse_scores)  # Convert back to RMSE
        
        cv_scores.append(cv_r2.mean())
        cv_rmse.append(cv_rmse_scores.mean())
        
        print(f"Degree {degree}: CV R² = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
        print(f"Degree {degree}: CV RMSE = {cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}")
    
    # Plot cross-validation results
    degrees = range(1, max_degree + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(degrees, cv_scores, 'o-', color='green', linewidth=2, markersize=8)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Cross-Validation R²')
    plt.title('Cross-Validation R² vs Polynomial Degree')
    plt.grid(True, alpha=0.3)
    plt.xticks(degrees)
    
    plt.subplot(1, 2, 2)
    plt.plot(degrees, cv_rmse, 's-', color='orange', linewidth=2, markersize=8)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Cross-Validation RMSE')
    plt.title('Cross-Validation RMSE vs Polynomial Degree')
    plt.grid(True, alpha=0.3)
    plt.xticks(degrees)
    
    plt.tight_layout()
    plt.savefig('plots/polynomial_cross_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal degree
    optimal_degree_cv_r2 = degrees[np.argmax(cv_scores)]
    optimal_degree_cv_rmse = degrees[np.argmin(cv_rmse)]
    
    print(f"\nOptimal degree (CV R²): {optimal_degree_cv_r2}")
    print(f"Optimal degree (CV RMSE): {optimal_degree_cv_rmse}")
    
    return optimal_degree_cv_r2, optimal_degree_cv_rmse, cv_scores, cv_rmse

def plot_polynomial_fits(models, X_train, X_test, y_train, y_test, optimal_degree):
    """Plot polynomial fits for different degrees"""
    print(f"\n=== Polynomial Fits Visualization (Optimal Degree: {optimal_degree}) ===")
    
    # Create fine grid for plotting
    X_plot = np.linspace(X_train.min(), X_train.max(), 1000).reshape(-1, 1)
    
    plt.figure(figsize=(15, 10))
    
    # Plot different degrees
    degrees_to_plot = [1, 2, 3, optimal_degree]
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, degree in enumerate(degrees_to_plot):
        if degree in models:
            plt.subplot(2, 2, i+1)
            
            # Plot training data
            plt.scatter(X_train, y_train, alpha=0.6, color='gray', label='Training data')
            
            # Plot test data
            plt.scatter(X_test, y_test, alpha=0.6, color='black', label='Test data')
            
            # Plot polynomial fit
            y_plot = models[degree].predict(X_plot)
            plt.plot(X_plot, y_plot, color=colors[i], linewidth=2, 
                    label=f'Degree {degree} polynomial')
            
            plt.xlabel('Median Income')
            plt.ylabel('Median House Value ($100,000s)')
            plt.title(f'Polynomial Regression (Degree {degree})')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/polynomial_fits_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_model_selection_methods(optimal_degree_r2, optimal_degree_rmse, optimal_degree_cv_r2, optimal_degree_cv_rmse):
    """Compare different model selection methods"""
    print("\n=== Model Selection Comparison ===")
    
    comparison_data = {
        'Method': ['Test R²', 'Test RMSE', 'CV R²', 'CV RMSE'],
        'Optimal Degree': [optimal_degree_r2, optimal_degree_rmse, optimal_degree_cv_r2, optimal_degree_cv_rmse]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Plot comparison
    methods = comparison_data['Method']
    degrees = comparison_data['Optimal Degree']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, degrees, color=['blue', 'red', 'green', 'orange'], alpha=0.7)
    plt.xlabel('Model Selection Method')
    plt.ylabel('Optimal Polynomial Degree')
    plt.title('Comparison of Model Selection Methods')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, degree in zip(bars, degrees):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(degree), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/model_selection_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run polynomial regression analysis"""
    print("Polynomial Regression Analysis - California Housing Dataset")
    print("=" * 60)
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Fit polynomial models
    models, train_scores, test_scores, train_rmse, test_rmse, X_train, X_test, y_train, y_test = fit_polynomial_models(X, y)
    
    # Analyze overfitting
    optimal_degree_r2, optimal_degree_rmse = plot_overfitting_analysis(train_scores, test_scores, train_rmse, test_rmse)
    
    # Cross-validation analysis
    optimal_degree_cv_r2, optimal_degree_cv_rmse, cv_scores, cv_rmse = cross_validation_analysis(X, y)
    
    # Plot polynomial fits
    plot_polynomial_fits(models, X_train, X_test, y_train, y_test, optimal_degree_cv_r2)
    
    # Compare model selection methods
    compare_model_selection_methods(optimal_degree_r2, optimal_degree_rmse, optimal_degree_cv_r2, optimal_degree_cv_rmse)
    
    print("\n=== Summary ===")
    print(f"Best polynomial degree (Cross-validation): {optimal_degree_cv_r2}")
    print(f"Cross-validation R²: {cv_scores[optimal_degree_cv_r2-1]:.4f}")
    print(f"Cross-validation RMSE: {cv_rmse[optimal_degree_cv_r2-1]:.4f}")
    print("All plots saved in the 'plots' directory.")
    print("Analysis complete!")

if __name__ == "__main__":
    main() 