
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utility_based_classifier import UtilityBasedCancerPredictor

def run_model_v1():
    # Load dataset for Model V1
    data = pd.read_csv("Datasets/breast_cancer.csv")
    data['diagnosis'] = (data['diagnosis'] == 'M').astype(int)
    feature_cols = [col for col in data.columns if col not in ['id', 'diagnosis']]
    X = data[feature_cols]
    y = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Train and evaluate Model V1
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Use UtilityBasedCancerPredictor for evaluation
    predictor = UtilityBasedCancerPredictor()
    results = predictor.evaluate_model(y_test, y_pred, model_name="Decision Tree V1")
    predictor.print_evaluation(results)
    return results

def run_model_v2():
    # Load dataset for Model V2
    data = pd.read_csv("Datasets/breast_cancer.csv")
    data['diagnosis'] = (data['diagnosis'] == 'M').astype(int)
    feature_cols = [col for col in data.columns if col not in ['id', 'diagnosis']]
    X = data[feature_cols]
    y = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Train and evaluate Model V2
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Use UtilityBasedCancerPredictor for evaluation
    predictor = UtilityBasedCancerPredictor()
    results = predictor.evaluate_model(y_test, y_pred, model_name="Decision Tree V2")
    predictor.print_evaluation(results)
    return results

def run_model_predictor():
    # Load and prepare data for the main breast_cancer_predictor model
    data = pd.read_csv("Datasets/breast_cancer.csv")
    data['diagnosis'] = (data['diagnosis'] == 'M').astype(int)
    feature_cols = [col for col in data.columns if col not in ['id', 'diagnosis']]
    X = data[feature_cols]
    y = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Custom utility matrix
    utility_matrix = {
        (0,0): 1.0,    # True Negative
        (0,1): -2.0,   # False Positive
        (1,0): -10.0,  # False Negative
        (1,1): 5.0     # True Positive
    }

    # Initialize and train model using the existing utility-based predictor
    predictor = UtilityBasedCancerPredictor(utility_matrix)
    best_model, results = predictor.train_and_optimize(X_train, X_test, y_train, y_test)
    predictor.print_evaluation(results)
    return results

if __name__ == "__main__":
    print("\n=== Running Model V1 ===")
    results_v1 = run_model_v1()
    
    print("\n=== Running Model V2 ===")
    results_v2 = run_model_v2()
    
    print("\n=== Running Main Predictor ===")
    results_predictor = run_model_predictor()

    # Comparing results
    print("\n=== Summary of Model Comparisons ===")
    print(f"Model V1 - Total Utility: {results_v1['total_utility']:.2f}, Average Utility: {results_v1['average_utility']:.2f}")
    print(f"Model V2 - Total Utility: {results_v2['total_utility']:.2f}, Average Utility: {results_v2['average_utility']:.2f}")
    print(f"Main Predictor - Total Utility: {results_predictor['total_utility']:.2f}, Average Utility: {results_predictor['average_utility']:.2f}")
