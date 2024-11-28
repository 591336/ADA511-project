import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("Datasets/breast_cancer.csv")
data['diagnosis'] = (data['diagnosis'] == 'M').astype(int)

# Select features
feature_cols = [col for col in data.columns if col not in ['id', 'diagnosis']]
X = data[feature_cols]
y = data['diagnosis']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
clf = clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

def evaluate_and_plot_confusion_matrix(y_true, y_pred, model_name="Decision Tree Custom"):
    """Function to evaluate the model and plot the confusion matrix."""
    cm = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Print evaluation results
    print(f"\n=== {model_name} Performance ===")
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(metrics.classification_report(y_true, y_pred))
    
    print("\nConfusion Matrix:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")

    # # Plot the confusion matrix
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
    #             xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.show()

# Evaluate and plot results
evaluate_and_plot_confusion_matrix(y_test, y_pred)
