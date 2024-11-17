import pandas as pd
from sklearn.model_selection import train_test_split
from utility_based_classification.utility_based_classifier import UtilityBasedCancerPredictor

# run utility based classifier on dataset

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
    
    # Custom utility matrix (optional customization)
    utility_matrix = {
        (0,0): 1.0,    # True Negative
        (0,1): -2.0,   # False Positive
        (1,0): -10.0,  # False Negative
        (1,1): 5.0     # True Positive
    }
    
    # Initialize and train model
    predictor = UtilityBasedCancerPredictor(utility_matrix)
    best_model, results = predictor.train_and_optimize(X_train, X_test, y_train, y_test)
    
    # Print evaluation results (including confusion matrix plot)
    predictor.print_evaluation(results)
