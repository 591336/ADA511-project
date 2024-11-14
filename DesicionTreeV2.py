import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load dataset
data = pd.read_csv("breast_cancer.csv")
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

# Calculate and display feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': clf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("=== Model Performance Metrics ===")
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")

print("\n=== Classification Report ===")
print(metrics.classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
print("\nConfusion Matrix Explanation:")
print(f"True Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

print("\n=== Top 10 Most Important Features ===")
print(feature_importance.head(10).to_string())

# Calculate some basic statistics about the dataset
print("\n=== Dataset Statistics ===")
print(f"Total samples: {len(data)}")
print(f"Malignant samples: {sum(data['diagnosis'] == 1)}")
print(f"Benign samples: {sum(data['diagnosis'] == 0)}")
print(f"Malignant percentage: {(sum(data['diagnosis'] == 1) / len(data) * 100):.1f}%")