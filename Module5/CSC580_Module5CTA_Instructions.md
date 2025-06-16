# Option #2: Building a Random Forest Classifier
In this assignment, you will use the iris datasetLinks to an external site. to classify iris plants based on measurements of their petal widths and sepal lengths. The dataset contains four variables measuring various parts of iris flowers of three related species and a fourth variable with the species name.

## Step 1: Load the data.

### Load the library with the iris dataset
from sklearn.datasets import load_iris
### Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
### Load pandas
import pandas as pd
### Load numpy
import numpy as np
### Set random seed
np.random.seed(0)
### Create an object called iris with the iris data
iris = load_iris()
### Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)
### View the top 5 rows
df.head()
### Add a new column with the species names; this is what we are going to try to predict
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
### View the top 5 rows
df.head()


Make a screenshot of the head of the dataset.


## Step 2: Create training and test data.

### Create a new column that:
for each row, generates a random number between 0 and 1, and
if that value is less than or equal to .75, then sets the value of that cell as True
and false otherwise. This is a quick and dirty way of randomly assigning some rows to
be used as the training data and some as the test data.

df['is_train'] = np.random.uniform(0, 1, len(df)) = .75
 
### View the top 5 rows
df.head()
### Create two new dataframes, one with the training rows and one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]
### Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))


Make a screenshot of the outputs.

 

## Step 3: Preprocess the data.

### Create a list of the feature column's names
features = df.columns[:4]
 
### View features
features

### train['species'] contains the actual species names. Before we can use it,
we need to convert each species name into a digit. So, in this case, there
are three species, which have been coded as 0, 1, or 2.

y = pd.factorize(train['species'])[0]
 
### View target
y


Make a screenshot of the outputs.

 

## Step 4: Train the random forest classifier.

### Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)
 
### Train the Classifier to take the training features and learn how they relate
### to the training y (the species)
clf.fit(train[features], y)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
            verbose=0, warm_start=False)
  
## Step 5: Apply the classifier to the test data and make a screenshot of the predicted probabilities of the first 10 observations.

### Apply the Classifier we trained to the test data (which, remember, it has never seen before)
clf.predict(test[features])

## Step 6: Evaluate the classifier by comparing the predicted and actual species for the first five observations.

## Step 7: Create a confusion matrix and use it to interpret the classification method.

Supply a screenshot of the confusion matrix.

### Create confusion matrix
pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])

## Step 8: View the list of features and their importance scores.

### View a list of the features and their importance scores
list(zip(train[features], clf.feature_importances_))