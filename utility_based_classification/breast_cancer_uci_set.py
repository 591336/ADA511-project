import pandas as pd
from sklearn.model_selection import train_test_split
from utility_based_classification.utility_based_classifier import UtilityBasedCancerPredictor

# Load the new Kaggle dataset
data = pd.read_csv('Datasets/breast_cancer_data_kaggle.csv')

# Data Preprocessing
if 'id' in data.columns:
    data = data.drop(columns=['id'])

# Convert 'diagnosis' to binary if it is not already (e.g., M=1, B=0)
data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Feature and target selection
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Handle any missing data
X = X.fillna(X.mean())

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model using the existing utility-based classifier
predictor = UtilityBasedCancerPredictor()
best_model, results = predictor.train_and_optimize(X_train, X_test, y_train, y_test)

# Print evaluation results (including confusion matrix plot)
predictor.print_evaluation(results)
