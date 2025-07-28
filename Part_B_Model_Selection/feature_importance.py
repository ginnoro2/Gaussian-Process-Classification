import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load California housing dataset and prepare for feature importance analysis"""
    print("=== Loading California Housing Dataset for Feature Importance Analysis ===")
    
    # Load dataset
    housing = fetch_california_housing()
    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    data['target'] = housing.target
    
    # Select features for analysis
    X = data.drop('target', axis=1)
    y = data['target']
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Features: {list(X.columns)}")
    
    return X, y

def lasso_feature_selection(X, y):
    """Use Lasso for feature selection"""
    print("\n=== Lasso Feature Selection ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try different alpha values
    alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    lasso_results = {}
    
    for alpha in alpha_values:
        print(f"\nFitting Lasso with alpha = {alpha}")
        
        # Fit Lasso
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = lasso.predict(X_train_scaled)
        y_test_pred = lasso.predict(X_test_scaled)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Count non-zero coefficients
        non_zero_coef = np.sum(lasso.coef_ != 0)
        
        lasso_results[alpha] = {
            'model': lasso,
            'coefficients': lasso.coef_,
            'intercept': lasso.intercept_,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'non_zero_coef': non_zero_coef
        }
        
        print(f"  Non-zero coefficients: {non_zero_coef}")
        print(f"  Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    return lasso_results, X_train_scaled, X_test_scaled, y_train, y_test

def random_forest_feature_importance(X, y):
    """Use Random Forest for feature importance analysis"""
    print("\n=== Random Forest Feature Importance ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Feature importance
    feature_importance = rf.feature_importances_
    
    rf_results = {
        'model': rf,
        'feature_importance': feature_importance,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
    
    print(f"Random Forest Performance:")
    print(f"  Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    return rf_results, X_train, X_test, y_train, y_test

def plot_lasso_analysis(lasso_results, feature_names):
    """Plot Lasso analysis results"""
    print("\n=== Lasso Analysis Visualization ===")
    
    alpha_values = list(lasso_results.keys())
    
    plt.figure(figsize=(15, 10))
    
    # Coefficient paths
    plt.subplot(2, 3, 1)
    coef_matrix = np.array([lasso_results[alpha]['coefficients'] for alpha in alpha_values])
    
    for i, feature in enumerate(feature_names):
        plt.plot(alpha_values, coef_matrix[:, i], marker='o', label=feature, linewidth=2)
    
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficient Value')
    plt.title('Lasso Coefficient Paths')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Performance metrics
    plt.subplot(2, 3, 2)
    train_rmse = [lasso_results[alpha]['train_rmse'] for alpha in alpha_values]
    test_rmse = [lasso_results[alpha]['test_rmse'] for alpha in alpha_values]
    
    plt.plot(alpha_values, train_rmse, 'o-', label='Train RMSE', linewidth=2, markersize=8)
    plt.plot(alpha_values, test_rmse, 's-', label='Test RMSE', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('RMSE')
    plt.title('Lasso Performance vs Alpha')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # R² scores
    plt.subplot(2, 3, 3)
    train_r2 = [lasso_results[alpha]['train_r2'] for alpha in alpha_values]
    test_r2 = [lasso_results[alpha]['test_r2'] for alpha in alpha_values]
    
    plt.plot(alpha_values, train_r2, 'o-', label='Train R²', linewidth=2, markersize=8)
    plt.plot(alpha_values, test_r2, 's-', label='Test R²', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('R² Score')
    plt.title('Lasso R² vs Alpha')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Number of non-zero coefficients
    plt.subplot(2, 3, 4)
    non_zero_counts = [lasso_results[alpha]['non_zero_coef'] for alpha in alpha_values]
    
    plt.plot(alpha_values, non_zero_counts, 'o-', color='purple', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Number of Non-Zero Coefficients')
    plt.title('Feature Sparsity vs Alpha')
    plt.grid(True, alpha=0.3)
    
    # Coefficient comparison for selected alpha
    plt.subplot(2, 3, 5)
    selected_alpha = 0.1  # Choose a reasonable alpha
    if selected_alpha in lasso_results:
        coef = lasso_results[selected_alpha]['coefficients']
        colors = ['red' if c == 0 else 'blue' for c in coef]
        
        bars = plt.bar(feature_names, np.abs(coef), color=colors, alpha=0.7)
        plt.xlabel('Features')
        plt.ylabel('|Coefficient|')
        plt.title(f'Lasso Coefficients (α={selected_alpha})')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, np.abs(coef)):
            if value > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Performance comparison
    plt.subplot(2, 3, 6)
    x = np.arange(len(alpha_values))
    width = 0.35
    
    plt.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.7)
    plt.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.7)
    plt.xlabel('Alpha Values')
    plt.ylabel('R² Score')
    plt.title('Train vs Test R²')
    plt.xticks(x, [f'{a:.3f}' for a in alpha_values], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/lasso_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_random_forest_importance(rf_results, feature_names):
    """Plot Random Forest feature importance"""
    print("\n=== Random Forest Feature Importance Visualization ===")
    
    feature_importance = rf_results['feature_importance']
    
    # Sort features by importance
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importance = feature_importance[sorted_indices]
    
    plt.figure(figsize=(12, 8))
    
    # Feature importance bar plot
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(sorted_features)), sorted_importance, color='skyblue', alpha=0.7)
    plt.xlabel('Features')
    plt.ylabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, importance in zip(bars, sorted_importance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{importance:.3f}', ha='center', va='bottom')
    
    # Cumulative importance
    plt.subplot(2, 2, 2)
    cumulative_importance = np.cumsum(sorted_importance)
    plt.plot(range(1, len(sorted_features) + 1), cumulative_importance, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance')
    plt.grid(True, alpha=0.3)
    
    # Feature importance pie chart
    plt.subplot(2, 2, 3)
    plt.pie(sorted_importance, labels=sorted_features, autopct='%1.1f%%', startangle=90)
    plt.title('Feature Importance Distribution')
    
    # Performance metrics
    plt.subplot(2, 2, 4)
    metrics = ['Train RMSE', 'Test RMSE', 'Train R²', 'Test R²']
    values = [rf_results['train_rmse'], rf_results['test_rmse'], 
              rf_results['train_r2'], rf_results['test_r2']]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylabel('Value')
    plt.title('Random Forest Performance Metrics')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/random_forest_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_feature_importance_methods(lasso_results, rf_results, feature_names):
    """Compare Lasso and Random Forest feature importance methods"""
    print("\n=== Feature Importance Methods Comparison ===")
    
    # Select a reasonable alpha for Lasso comparison
    selected_alpha = 0.1
    if selected_alpha in lasso_results:
        lasso_coef = np.abs(lasso_results[selected_alpha]['coefficients'])
        rf_importance = rf_results['feature_importance']
        
        # Normalize both to [0, 1] for comparison
        lasso_normalized = lasso_coef / np.max(lasso_coef) if np.max(lasso_coef) > 0 else lasso_coef
        rf_normalized = rf_importance / np.max(rf_importance)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Feature': feature_names,
            'Lasso (Normalized)': lasso_normalized,
            'Random Forest (Normalized)': rf_normalized
        })
        
        # Sort by Random Forest importance
        comparison_df = comparison_df.sort_values('Random Forest (Normalized)', ascending=False)
        
        print("Feature Importance Comparison (Normalized):")
        print(comparison_df.to_string(index=False))
        
        # Plot comparison
        plt.figure(figsize=(15, 8))
        
        # Side-by-side bar plot
        plt.subplot(1, 2, 1)
        x = np.arange(len(comparison_df))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, comparison_df['Lasso (Normalized)'], 
                        width, label='Lasso', alpha=0.7, color='blue')
        bars2 = plt.bar(x + width/2, comparison_df['Random Forest (Normalized)'], 
                        width, label='Random Forest', alpha=0.7, color='orange')
        
        plt.xlabel('Features')
        plt.ylabel('Normalized Importance')
        plt.title('Feature Importance Comparison')
        plt.xticks(x, comparison_df['Feature'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Correlation plot
        plt.subplot(1, 2, 2)
        plt.scatter(comparison_df['Lasso (Normalized)'], 
                   comparison_df['Random Forest (Normalized)'], 
                   alpha=0.7, s=100)
        
        # Add feature labels
        for i, feature in enumerate(comparison_df['Feature']):
            plt.annotate(feature, 
                        (comparison_df['Lasso (Normalized)'].iloc[i], 
                         comparison_df['Random Forest (Normalized)'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Lasso Importance (Normalized)')
        plt.ylabel('Random Forest Importance (Normalized)')
        plt.title('Correlation between Methods')
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(comparison_df['Lasso (Normalized)'], 
                                 comparison_df['Random Forest (Normalized)'])[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('plots/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nCorrelation between Lasso and Random Forest importance: {correlation:.3f}")
        
        return comparison_df, correlation

def main():
    """Main function to run feature importance analysis"""
    print("Feature Importance Analysis - California Housing Dataset")
    print("=" * 60)
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    feature_names = list(X.columns)
    
    # Lasso feature selection
    lasso_results, X_train_scaled, X_test_scaled, y_train, y_test = lasso_feature_selection(X, y)
    
    # Random Forest feature importance
    rf_results, X_train_rf, X_test_rf, y_train_rf, y_test_rf = random_forest_feature_importance(X, y)
    
    # Plot Lasso analysis
    plot_lasso_analysis(lasso_results, feature_names)
    
    # Plot Random Forest importance
    plot_random_forest_importance(rf_results, feature_names)
    
    # Compare methods
    comparison_df, correlation = compare_feature_importance_methods(lasso_results, rf_results, feature_names)
    
    print("\n=== Summary ===")
    print("Feature importance analysis completed!")
    print(f"Correlation between methods: {correlation:.3f}")
    print("All plots saved in the 'plots' directory.")
    print("Analysis complete!")

if __name__ == "__main__":
    main() 