import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load California housing dataset and perform initial exploration"""
    print("=== Loading and Exploring California Housing Dataset ===")
    
    # Load dataset
    housing = fetch_california_housing()
    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    data['target'] = housing.target
    
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {list(data.columns)}")
    print(f"Target variable: Median house value (in $100,000s)")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(data.describe())
    
    # Check for missing values
    print(f"\nMissing values:\n{data.isnull().sum()}")
    
    return data

def exploratory_data_analysis(data):
    """Perform comprehensive EDA"""
    print("\n=== Exploratory Data Analysis ===")
    
    # Correlation analysis
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Distribution of target variable
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(data['target'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Median House Value ($100,000s)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Target Variable')
    plt.grid(True, alpha=0.3)
    
    # Box plot of target by different features
    plt.subplot(1, 3, 2)
    data['income_cat'] = pd.cut(data['MedInc'], bins=5, labels=['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])
    sns.boxplot(x='income_cat', y='target', data=data)
    plt.xlabel('Income Category')
    plt.ylabel('Median House Value ($100,000s)')
    plt.title('House Values by Income Category')
    plt.xticks(rotation=45)
    
    # Scatter plot of most correlated feature
    plt.subplot(1, 3, 3)
    plt.scatter(data['MedInc'], data['target'], alpha=0.6, color='green')
    plt.xlabel('Median Income')
    plt.ylabel('Median House Value ($100,000s)')
    plt.title('House Values vs Median Income')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/eda_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Remove the temporary column
    data.drop('income_cat', axis=1, inplace=True)
    
    return correlation_matrix

def feature_selection(data, correlation_matrix):
    """Perform feature selection based on correlation and statistical tests"""
    print("\n=== Feature Selection ===")
    
    # Select features with highest correlation to target
    target_correlations = correlation_matrix['target'].abs().sort_values(ascending=False)
    print("Feature correlations with target:")
    print(target_correlations)
    
    # Select top 5 features
    top_features = target_correlations[1:6].index.tolist()  # Exclude target itself
    print(f"\nSelected features: {top_features}")
    
    # Statistical feature selection
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Use f_regression for feature selection
    selector = SelectKBest(score_func=f_regression, k=5)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"Statistically selected features: {selected_features}")
    
    return selected_features

def fit_linear_regression(data, selected_features):
    """Fit linear regression model"""
    print("\n=== Fitting Linear Regression Model ===")
    
    X = data[selected_features]
    y = data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Model coefficients
    coefficients = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': model.coef_
    })
    coefficients = coefficients.sort_values('Coefficient', key=abs, ascending=False)
    
    print("Model Coefficients:")
    print(coefficients)
    print(f"Intercept: {model.intercept_:.4f}")
    
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_train_pred, y_test_pred

def evaluate_model(y_train, y_test, y_train_pred, y_test_pred):
    """Evaluate model performance"""
    print("\n=== Model Evaluation ===")
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    return {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae
    }

def plot_residuals(X_train_scaled, y_train, y_train_pred, X_test_scaled, y_test, y_test_pred):
    """Plot residuals and interpret results"""
    print("\n=== Residual Analysis ===")
    
    # Calculate residuals
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    plt.figure(figsize=(15, 10))
    
    # Residuals vs Predicted (Training)
    plt.subplot(2, 3, 1)
    plt.scatter(y_train_pred, train_residuals, alpha=0.6, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted (Training)')
    plt.grid(True, alpha=0.3)
    
    # Residuals vs Predicted (Test)
    plt.subplot(2, 3, 2)
    plt.scatter(y_test_pred, test_residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted (Test)')
    plt.grid(True, alpha=0.3)
    
    # Residuals histogram
    plt.subplot(2, 3, 3)
    plt.hist(train_residuals, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Training')
    plt.hist(test_residuals, bins=30, alpha=0.7, color='green', edgecolor='black', label='Test')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot
    plt.subplot(2, 3, 4)
    stats.probplot(train_residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Training Residuals)')
    
    # Residuals vs features
    plt.subplot(2, 3, 5)
    plt.scatter(X_train_scaled[:, 0], train_residuals, alpha=0.6, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('First Feature (Scaled)')
    plt.ylabel('Residuals')
    plt.title('Residuals vs First Feature')
    plt.grid(True, alpha=0.3)
    
    # Actual vs Predicted
    plt.subplot(2, 3, 6)
    plt.scatter(y_train, y_train_pred, alpha=0.6, color='blue', label='Training')
    plt.scatter(y_test, y_test_pred, alpha=0.6, color='green', label='Test')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def verify_assumptions(train_residuals, test_residuals):
    """Verify linear regression assumptions"""
    print("\n=== Linear Regression Assumptions Verification ===")
    
    # 1. Linearity (already checked in residual plots)
    print("1. Linearity: Checked via residual plots")
    
    # 2. Independence (assumed for this dataset)
    print("2. Independence: Assumed for housing data")
    
    # 3. Homoscedasticity
    print("\n3. Homoscedasticity Test:")
    # Levene's test for homoscedasticity
    from scipy.stats import levene
    # Split residuals into groups for testing
    train_res_1 = train_residuals[:len(train_residuals)//2]
    train_res_2 = train_residuals[len(train_residuals)//2:]
    stat, p_value = levene(train_res_1, train_res_2)
    print(f"Levene's test p-value: {p_value:.4f}")
    print(f"Homoscedasticity assumption: {'✓' if p_value > 0.05 else '✗'}")
    
    # 4. Normality
    print("\n4. Normality Test:")
    from scipy.stats import shapiro
    stat, p_value = shapiro(train_residuals)
    print(f"Shapiro-Wilk test p-value: {p_value:.4f}")
    print(f"Normality assumption: {'✓' if p_value > 0.05 else '✗'}")
    
    # 5. Multicollinearity
    print("\n5. Multicollinearity: Checked via correlation matrix")
    
    return {
        'homoscedasticity_p': p_value,
        'normality_p': p_value
    }

def main():
    """Main function to run linear regression analysis"""
    print("Linear Regression Analysis - California Housing Dataset")
    print("=" * 60)
    
    # Load and explore data
    data = load_and_explore_data()
    
    # Perform EDA
    correlation_matrix = exploratory_data_analysis(data)
    
    # Feature selection
    selected_features = feature_selection(data, correlation_matrix)
    
    # Fit model
    model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_train_pred, y_test_pred = fit_linear_regression(data, selected_features)
    
    # Evaluate model
    metrics = evaluate_model(y_train, y_test, y_train_pred, y_test_pred)
    
    # Plot residuals
    plot_residuals(X_train_scaled, y_train, y_train_pred, X_test_scaled, y_test, y_test_pred)
    
    # Verify assumptions
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    assumptions = verify_assumptions(train_residuals, test_residuals)
    
    print("\n=== Summary ===")
    print(f"Model Performance: R² = {metrics['test_r2']:.4f}, RMSE = {metrics['test_rmse']:.4f}")
    print("All plots saved in the 'plots' directory.")
    print("Analysis complete!")

if __name__ == "__main__":
    main() 