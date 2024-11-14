import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from utility_based_classifier import UtilityBasedCancerPredictor

# first test directly taken form datacamp

# Load dataset
data = pd.read_csv("Datasets/breast_cancer.csv")
data['diagnosis'] = (data['diagnosis'] == 'M').astype(int)

# Select features
feature_cols = [col for col in data.columns if col not in ['id', 'diagnosis']]
X = data[feature_cols]
y = data['diagnosis']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train model
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
clf = clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Use UtilityBasedCancerPredictor for consistent evaluation and visualization
predictor = UtilityBasedCancerPredictor()
results = predictor.evaluate_model(y_test, y_pred, model_name="Decision Tree V1")
predictor.print_evaluation(results)
