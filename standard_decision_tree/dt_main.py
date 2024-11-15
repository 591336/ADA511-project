import pandas as pd
from sklearn.model_selection import train_test_split
from DecisionTreeV1 import StandardCancerPredictor

if __name__ == "__main__":
    # Load and prepare data
    data = pd.read_csv("Datasets/breast_cancer.csv")
    data['diagnosis'] = (data['diagnosis'] == 'M').astype(int)  # 1 for Malignant, 0 for Benign
    
    # Select features
    feature_cols = [col for col in data.columns if col not in ['id', 'diagnosis']]
    X = data[feature_cols]
    y = data['diagnosis']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Initialize and train standard model
    standard_predictor = StandardCancerPredictor()
    best_model_std, results_std = standard_predictor.train_and_optimize(X_train, X_test, y_train, y_test)
    
    # Print evaluation results (including confusion matrix plot)
    standard_predictor.print_evaluation(results_std)
