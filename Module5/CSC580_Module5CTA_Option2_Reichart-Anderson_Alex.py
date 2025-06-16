# Module 5 Critical Thinking Assignment
# Option 2: Building a Random Forest Classifier

# Alexander Reichart-Anderson
# MS in AI and ML, Colorado State University Global
# CSC580-1: Applying Machine Learning & Neural Networks - Capstone
# Dr. Joseph Issa
# June 15, 2025

# Step 0: Import Required Packages
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
np.random.seed(0) # Set random seed for reproducibility

# Step 1: Load and prepare data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("Head of the dataset after loading:") ## ---------- # Screenshot 1: Head of Complete Data Set
print(df.head())
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print("\nHead of the dataset after adding species column:") ## ---------- # Screenshot 1: Head of Data Set with Species
print(df.head())

# Step 2: Create training/test split
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75
print("\nHead of the dataset after adding is_train column:") ## ---------- # Screenshot 2: Head of dataset with training column
print(df.head())
train, test = df[df['is_train']], df[~df['is_train']]
print("\nNumber of observations in the training data:", len(train)) ## ---------- # Screenshot 2: Length of training and test data set
print("Number of observations in the test data:", len(test))

# Step 3: Preprocess data
features = df.columns[:4]
print("\nFeatures used for training:") ## ---------- # Screenshot 3: Training Features
print(features)
y = pd.factorize(train['species'])[0]
print("\nEncoded target variable (y) for training data:") ## ---------- # Screenshot 3: Target Variable (first 10 for brevity)
print(y[:10])

# Step 4: Train classifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(train[features], y)

# Step 5: Predictions and probabilities
test_probs = clf.predict_proba(test[features])
print("\nPredicted probabilities (first 10):")
print(pd.DataFrame(test_probs[:10], columns=iris.target_names))

# Step 6: Predictions vs actual (first 5)
preds = clf.predict(test[features])
comparison = pd.DataFrame({'Predicted': pd.Categorical.from_codes(preds, iris.target_names),'Actual': test['species']})
print("\nFirst 5 predictions vs actual:")
print(comparison.head())

# Step 7: Confusion Matrix
conf_matrix = pd.crosstab(test['species'],pd.Categorical.from_codes(preds, iris.target_names),rownames=['Actual'],colnames=['Predicted'])
print("\nConfusion matrix:") ## ---------- # Screenshot 4: Confusion Matrix Screenshot
print(conf_matrix) 

# Step 8: Feature importances
importance = pd.DataFrame({'Feature': features,'Importance': clf.feature_importances_})
print("\nFeature importances:")
print(importance.sort_values('Importance', ascending=False))