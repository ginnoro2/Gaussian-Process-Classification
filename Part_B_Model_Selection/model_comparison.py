import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load California housing dataset and prepare for model comparison"""
    print("=== Loading California Housing Dataset for Model Comparison ===")
    
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

def fit_basic_models(X, y):
    """Fit basic models without hyperparameter tuning"""
    print("\n=== Fitting Basic Models ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nFitting {name}...")
        
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        results[name] = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
        
        print(f"{name}:")
        print(f"  Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"  Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    
    return results, X_train_scaled, X_test_scaled, y_train, y_test

def cross_validation_analysis(X, y):
    """Perform cross-validation for model comparison"""
    print("\n=== Cross-Validation Analysis ===")
    
    # Define models with default parameters
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    cv_results = {}
    
    for name, model in models.items():
        print(f"\nPerforming cross-validation for {name}...")
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        # Perform cross-validation
        cv_r2 = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        cv_rmse = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_rmse)  # Convert back to RMSE
        cv_mae = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_mae  # Convert back to MAE
        
        cv_results[name] = {
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std(),
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'cv_mae_mean': cv_mae.mean(),
            'cv_mae_std': cv_mae.std()
        }
        
        print(f"{name}:")
        print(f"  CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
        print(f"  CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
        print(f"  CV MAE: {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
    
    return cv_results

def hyperparameter_tuning(X, y):
    """Perform hyperparameter tuning using GridSearchCV and RandomizedSearchCV"""
    print("\n=== Hyperparameter Tuning ===")
    
    # Define parameter grids
    param_grids = {
        'Ridge Regression': {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        },
        'Lasso Regression': {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        },
        'Elastic Net': {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    }
    
    tuned_models = {}
    
    for name, param_grid in param_grids.items():
        print(f"\nTuning hyperparameters for {name}...")
        
        # Create pipeline
        if name == 'Ridge Regression':
            model = Ridge()
        elif name == 'Lasso Regression':
            model = Lasso()
        elif name == 'Elastic Net':
            model = ElasticNet()
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X, y)
        
        # Get best parameters and score
        best_params = grid_search.best_params_
        best_score = np.sqrt(-grid_search.best_score_)
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV RMSE: {best_score:.4f}")
        
        tuned_models[name] = {
            'best_model': grid_search.best_estimator_,
            'best_params': best_params,
            'best_cv_rmse': best_score
        }
    
    return tuned_models

def plot_model_comparison(results, cv_results):
    """Plot model comparison results"""
    print("\n=== Model Comparison Visualization ===")
    
    # Prepare data for plotting
    model_names = list(results.keys())
    test_rmse = [results[name]['test_rmse'] for name in model_names]
    test_r2 = [results[name]['test_r2'] for name in model_names]
    cv_rmse = [cv_results[name]['cv_rmse_mean'] for name in model_names]
    cv_r2 = [cv_results[name]['cv_r2_mean'] for name in model_names]
    
    plt.figure(figsize=(15, 10))
    
    # Test RMSE comparison
    plt.subplot(2, 3, 1)
    bars1 = plt.bar(model_names, test_rmse, color='skyblue', alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('Test RMSE')
    plt.title('Test RMSE Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, test_rmse):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom')
    
    # Test R² comparison
    plt.subplot(2, 3, 2)
    bars2 = plt.bar(model_names, test_r2, color='lightgreen', alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('Test R²')
    plt.title('Test R² Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, test_r2):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    # CV RMSE comparison
    plt.subplot(2, 3, 3)
    bars3 = plt.bar(model_names, cv_rmse, color='orange', alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('CV RMSE')
    plt.title('Cross-Validation RMSE Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars3, cv_rmse):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom')
    
    # CV R² comparison
    plt.subplot(2, 3, 4)
    bars4 = plt.bar(model_names, cv_r2, color='pink', alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('CV R²')
    plt.title('Cross-Validation R² Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars4, cv_r2):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    # Combined metrics comparison
    plt.subplot(2, 3, 5)
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, test_rmse, width, label='Test RMSE', alpha=0.7)
    plt.bar(x + width/2, cv_rmse, width, label='CV RMSE', alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('RMSE')
    plt.title('Test vs CV RMSE Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Model coefficients comparison
    plt.subplot(2, 3, 6)
    # Get coefficients for regularized models
    coef_data = []
    coef_labels = []
    
    for name in ['Ridge Regression', 'Lasso Regression', 'Elastic Net']:
        if name in results:
            model = results[name]['model']
            if hasattr(model, 'coef_'):
                coef_data.append(np.abs(model.coef_))
                coef_labels.append(name)
    
    if coef_data:
        coef_df = pd.DataFrame(coef_data, columns=[f'Feature_{i}' for i in range(len(coef_data[0]))])
        coef_df.index = coef_labels
        coef_df.T.plot(kind='bar', ax=plt.gca())
        plt.xlabel('Features')
        plt.ylabel('|Coefficient|')
        plt.title('Feature Importance (Regularized Models)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def select_best_model(results, cv_results, tuned_models):
    """Select the best model and explain reasoning"""
    print("\n=== Model Selection and Reasoning ===")
    
    # Create comparison table
    comparison_data = []
    
    for name in results.keys():
        comparison_data.append({
            'Model': name,
            'Test RMSE': results[name]['test_rmse'],
            'Test R²': results[name]['test_r2'],
            'CV RMSE': cv_results[name]['cv_rmse_mean'],
            'CV R²': cv_results[name]['cv_r2_mean']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('CV R²', ascending=False)
    
    print("Model Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Select best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_cv_r2 = comparison_df.iloc[0]['CV R²']
    best_cv_rmse = comparison_df.iloc[0]['CV RMSE']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best CV R²: {best_cv_r2:.4f}")
    print(f"Best CV RMSE: {best_cv_rmse:.4f}")
    
    # Reasoning
    print(f"\nReasoning for selecting {best_model_name}:")
    print("1. Cross-validation provides more robust performance estimation")
    print("2. Higher R² indicates better explanatory power")
    print("3. Lower RMSE indicates better predictive accuracy")
    
    if best_model_name in tuned_models:
        print(f"4. Hyperparameter tuning was performed for {best_model_name}")
        print(f"   Best parameters: {tuned_models[best_model_name]['best_params']}")
    
    return best_model_name, comparison_df

def main():
    """Main function to run model comparison analysis"""
    print("Model Selection and Evaluation - California Housing Dataset")
    print("=" * 60)
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Fit basic models
    results, X_train_scaled, X_test_scaled, y_train, y_test = fit_basic_models(X, y)
    
    # Cross-validation analysis
    cv_results = cross_validation_analysis(X, y)
    
    # Hyperparameter tuning
    tuned_models = hyperparameter_tuning(X, y)
    
    # Plot model comparison
    plot_model_comparison(results, cv_results)
    
    # Select best model
    best_model_name, comparison_df = select_best_model(results, cv_results, tuned_models)
    
    print("\n=== Summary ===")
    print(f"Best performing model: {best_model_name}")
    print("All plots saved in the 'plots' directory.")
    print("Analysis complete!")

if __name__ == "__main__":
    main() 