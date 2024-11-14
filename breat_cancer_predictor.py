# breast_cancer_predictor.py

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from utility_based_classifier import UtilityBasedCancerPredictor
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot a confusion matrix using seaborn."""
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

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
        X, y, test_size=0.3, # split size, meaning 30% test size, 70% training
        random_state=42 # Setting to a specific number ensures that the data is split into training and testing sets in the same way every run
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
    
    # Print evaluation results
    predictor.print_evaluation(results)
    
    # Visualize confusion matrix
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, labels=["Benign", "Malignant"])
