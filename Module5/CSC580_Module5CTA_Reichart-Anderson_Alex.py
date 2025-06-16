# Module 5 Critical Thinking Assignment
# Option 2: Building a Random Forest Classifier

# Alexander Reichart-Anderson
# MS in AI and ML, Colorado State University Global
# CSC580-1: Applying Machine Learning & Neural Networks - Capstone
# Dr. Joseph Issa
# June 15, 2025

# Step 0: 
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Set random seed for reproducibility
np.random.seed(0)

# Step 1: Load and prepare data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Step 2: Create training/test split
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75
train, test = df[df['is_train']], df[~df['is_train']]
print("Training samples:", len(train))
print("Test samples:", len(test))

# Step 3: Preprocess data
features = iris.feature_names
y_train = pd.factorize(train['species'])[0]

# Step 4: Train classifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(train[features], y_train)

# Step 5: Predictions and probabilities
test_probs = clf.predict_proba(test[features])
print("\nPredicted probabilities (first 10):")
print(pd.DataFrame(test_probs[:10], columns=iris.target_names))

# Step 6: Predictions vs actual (first 5)
preds = clf.predict(test[features])
comparison = pd.DataFrame({
    'Predicted': pd.Categorical.from_codes(preds, iris.target_names),
    'Actual': test['species']
})
print("\nFirst 5 predictions vs actual:")
print(comparison.head())

# Step 7: Confusion matrix
conf_matrix = pd.crosstab(
    test['species'],
    pd.Categorical.from_codes(preds, iris.target_names),
    rownames=['Actual'],
    colnames=['Predicted']
)
print("\nConfusion matrix:")
print(conf_matrix)

# Step 8: Feature importances
importance = pd.DataFrame({
    'Feature': features,
    'Importance': clf.feature_importances_
})

print("\nFeature importances:")
print(importance.sort_values('Importance', ascending=False))
