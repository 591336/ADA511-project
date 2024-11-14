import pandas as pd
from sklearn.model_selection import train_test_split
from utility_based_classifier import UtilityBasedCancerPredictor

# Load the new Kaggle dataset
data = pd.read_csv('Datasets/breast_cancer_data_kaggle.csv')

# Inspect the data structure (optional, can be removed later)
print(data.head())

# Data Preprocessing
# Example preprocessing - adjust as needed based on the actual data structure:
# - Drop any unnecessary columns (like IDs)
if 'id' in data.columns:
    data = data.drop(columns=['id'])

# Assume the 'diagnosis' column is the target (adjust if different)
# Convert 'diagnosis' to a binary format if it is not already (e.g., M=1, B=0)
if 'diagnosis' in data.columns:
    data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Feature and target selection
X = data.drop('diagnosis', axis=1)  # All columns except 'diagnosis'
y = data['diagnosis']  # Target variable

# Handle any missing data if necessary
X = X.fillna(X.mean())  # Simple mean imputation, modify as appropriate

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train model using the existing utility-based classifier
predictor = UtilityBasedCancerPredictor()
best_model, results = predictor.train_and_optimize(X_train, X_test, y_train, y_test)

# Print evaluation results
predictor.print_evaluation(results)
