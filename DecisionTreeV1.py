import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Set figure style without using seaborn's style
plt.style.use('default')
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['axes.grid'] = True

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

# Calculate feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': clf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# 1. Feature Importance Plot
top_10_features = feature_importance.head(10)
ax1.barh(np.arange(len(top_10_features)), top_10_features['importance'])
ax1.set_yticks(np.arange(len(top_10_features)))
ax1.set_yticklabels(top_10_features['feature'])
ax1.set_title('Top 10 Most Important Features')
ax1.set_xlabel('Feature Importance')

# 2. Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
im = ax2.imshow(cm, cmap='Blues')
ax2.set_title('Confusion Matrix')
ax2.set_ylabel('True Label')
ax2.set_xlabel('Predicted Label')
# Add text annotations to confusion matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax2.text(j, i, str(cm[i, j]), ha='center', va='center')

# 3. Distribution of Mean Radius by Diagnosis
benign = data[data['diagnosis'] == 0]['radius_mean']
malignant = data[data['diagnosis'] == 1]['radius_mean']
ax3.boxplot([benign, malignant], labels=['Benign', 'Malignant'])
ax3.set_title('Distribution of Mean Radius by Diagnosis')
ax3.set_ylabel('Radius Mean')

# 4. Correlation between top features
top_features = feature_importance['feature'].head(5).tolist()
correlation_matrix = data[top_features].corr()
im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
ax4.set_title('Correlation between Top 5 Features')
# Add text annotations to correlation matrix
for i in range(len(top_features)):
    for j in range(len(top_features)):
        ax4.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", 
                ha='center', va='center')
ax4.set_xticks(range(len(top_features)))
ax4.set_yticks(range(len(top_features)))
ax4.set_xticklabels(top_features, rotation=45)
ax4.set_yticklabels(top_features)

# Adjust layout
plt.tight_layout()

# Print model performance metrics
print("\nModel Performance Metrics:")
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(metrics.classification_report(y_test, y_pred))

# Save the plot
plt.savefig('breast_cancer_analysis.png', dpi=300, bbox_inches='tight')
plt.show()