import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from typing import Dict, Tuple

#implemented utility, now split in new class -> utlity_based_classifier

class UtilityBasedCancerPredictor:
    def __init__(self, utility_matrix: Dict[Tuple[int, int], float] = None):
        """
        Initialize the predictor with a utility matrix.
        
        The utility matrix should contain values for all possible (true_label, predicted_label) pairs:
        - (0,0): True Negative (correctly identifying benign)
        - (0,1): False Positive (incorrectly identifying as malignant)
        - (1,0): False Negative (incorrectly identifying as benign)
        - (1,1): True Positive (correctly identifying malignant)
        """
        # Default utility matrix if none provided
        self.utility_matrix = utility_matrix or {
            (0,0): 1.0,    # True Negative: Small positive utility
            (0,1): -2.0,   # False Positive: Moderate negative utility (unnecessary anxiety/procedures)
            (1,0): -10.0,  # False Negative: Very high negative utility (missed cancer)
            (1,1): 5.0     # True Positive: High positive utility (early detection)
        }
        
        self.best_model = None
        self.best_utility = float('-inf')
        
    def calculate_total_utility(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the total utility based on predictions and true values."""
        total_utility = 0
        for true_val, pred_val in zip(y_true, y_pred):
            total_utility += self.utility_matrix[(true_val, pred_val)]
        return total_utility
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
        """Evaluate model performance with detailed metrics and utility calculation."""
        # Calculate standard metrics
        cm = metrics.confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate utility-based metrics
        total_utility = self.calculate_total_utility(y_true, y_pred)
        average_utility = total_utility / len(y_true)
        
        # Calculate clinical metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive predictive value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
        
        results = {
            'model_name': model_name,
            'total_utility': total_utility,
            'average_utility': average_utility,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }
        
        return results
    
    def print_evaluation(self, results: dict):
        """Print formatted evaluation results."""
        print(f"\n=== {results['model_name']} Performance ===")
        print(f"Total Utility: {results['total_utility']:.2f}")
        print(f"Average Utility per Prediction: {results['average_utility']:.2f}")
        
        print("\nConfusion Matrix:")
        print(f"True Negatives: {results['true_negatives']}")
        print(f"False Positives: {results['false_positives']}")
        print(f"False Negatives: {results['false_negatives']}")
        print(f"True Positives: {results['true_positives']}")
        
        print("\nClinical Metrics:")
        print(f"Sensitivity (True Positive Rate): {results['sensitivity']:.4f}")
        print(f"Specificity (True Negative Rate): {results['specificity']:.4f}")
        print(f"Positive Predictive Value: {results['ppv']:.4f}")
        print(f"Negative Predictive Value: {results['npv']:.4f}")
    
    def custom_scorer(self, estimator, X, y):
        """Custom scorer for GridSearchCV that uses our utility function."""
        y_pred = estimator.predict(X)
        return self.calculate_total_utility(y, y_pred)
    
    def train_and_optimize(self, X_train, X_test, y_train, y_test):
        """Train and optimize the model using utility-based grid search."""
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5, 7, 9, 11],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [
                {0: 1, 1: 1},
                {0: 1, 1: 5},
                {0: 1, 1: 10},
                'balanced'
            ],
            'criterion': ['gini', 'entropy']
        }
        
        # Initialize base model
        base_model = DecisionTreeClassifier(random_state=42)
        
        # Perform grid search with utility-based scoring
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=self.custom_scorer,
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate best model
        self.best_model = grid_search.best_estimator_
        y_pred = self.best_model.predict(X_test)
        results = self.evaluate_model(y_test, y_pred, "Optimized Model")
        
        print("\n=== Best Parameters ===")
        print(grid_search.best_params_)
        self.print_evaluation(results)
        
        return self.best_model, results

# Usage example
if __name__ == "__main__":
    # Load and prepare data
    data = pd.read_csv("Datasets/breast_cancer.csv")
    data['diagnosis'] = (data['diagnosis'] == 'M').astype(int)
    
    # Select features
    feature_cols = [col for col in data.columns if col not in ['id', 'diagnosis']]
    X = data[feature_cols]
    y = data['diagnosis']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Custom utility matrix - adjust these values based on medical expertise
    utility_matrix = {
        (0,0): 1.0,    # True Negative
        (0,1): -2.0,   # False Positive
        (1,0): -10.0,  # False Negative
        (1,1): 5.0     # True Positive
    }
    
    # Initialize and train model
    predictor = UtilityBasedCancerPredictor(utility_matrix)
    best_model, results = predictor.train_and_optimize(X_train, X_test, y_train, y_test)