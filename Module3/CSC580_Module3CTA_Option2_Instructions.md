# Option #2: Predicting Fuel Efficiency Using TensorFlow
In a regression problem, we aim to predict the output of a continuous value, like a price or a probability. Contrast this with a classification problem, where we aim to select a class from a list of classes (for example, where a picture contains an apple or an orange, recognizing which fruit is in the picture).

This assignment uses the classic Auto MPG (https://archive.ics.uci.edu/dataset/9/auto+mpg) Dataset and builds a model to predict the fuel efficiency of late-1970s and early-1980s automobiles. The data model includes descriptions of many automobiles from that time period. The description for an automobile includes attributes such as cylinders, displacement, horsepower, and weight.

For this project, you will use the tf.keras (https://www.tensorflow.org/api_docs/python/tf/keras) API. The Python packages you will need include the following:


Use seaborn for pairplot  !pip install -q seaborn    # Use some functions from Tensorflow_docs  !pip install -q git+https://github.com/tensorflow/docs    
from __future__ import absolute_import, division, print_function, unicode_literals    import pathlib    import matplotlib.pyplot as plt  import numpy as np  import pandas as pd  import seaborn as sns    
import tensorflow as tf    from tensorflow import keras  from tensorflow.keras import layers    print(tf.__version__)    
2.0.0
import tensorflow_docs as tfdocs  import tensorflow_docs.plots  import tensorflow_docs.modeling
 

## Step 1: Download the dataset using keras get_file method.
 

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")dataset_path 

Downloading data from http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data

32768/30286 [================================] - 0s 1us/step

'/home/kbuilder/.keras/datasets/auto-mpg.data'

## Step 2: Import database using Pandas.

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t',sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()dataset.tail()

## Step 3: Take a screenshot of the tail of the dataset. ------------------------------
 

## Step 4: Split the data into train and test.
train_dataset = dataset.sample(frac=0.8,random_state=0)  test_dataset = dataset.drop(train_dataset.index)    

## Step 5: Inspect the data.
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

## Step 6: Take a screenshot of the tail of the plots. ------------------------------

## Step 7: Review the statistics.
train_stats = train_dataset.describe()  train_stats.pop("MPG")  train_stats = train_stats.transpose()  train_stats

## Step 8: Take a screenshot of the tail of the statistics. ------------------------------
 

## Step 9: Split features from labels.

## Step 10: Separate the target value, or "label," from the features.
This label is the value that you will train the model to predict.
train_labels = train_dataset.pop('MPG')  test_labels = test_dataset.pop('MPG')    
 
## Step 11: Normalize the data.
It is good practice to normalize features that use different scales and ranges. Although the model might converge without feature normalization, it makes training more difficult, and it makes the resulting model dependent on the choice of units used in the input.

def norm(x):    return (x - train_stats['mean']) / train_stats['std']  normed_train_data = norm(train_dataset)  normed_test_data = norm(test_dataset)    
This normalized data is what you will use to train the model.

## Step 12: Build the model.
Use a Sequential model with two densely connected hidden layers and an output layer that returns a single, continuous value. The model building steps are wrapped in a function, build_model.


def build_model():    model = keras.Sequential([      layers.Dense(64, activation='relu',       input_shape=[len(train_dataset.keys())]),      layers.Dense(64, activation='relu'),      layers.Dense(1)    ])      optimizer = tf.keras.optimizers.RMSprop(0.001)      model.compile(loss='mse',                  optimizer=optimizer,                  metrics=['mae', 'mse'])    return model    
model = build_model()
 

## Step 13: Inspect the model.

Use the .summary method to print a simple description of the model.
           model           .           summary           ()                       

## Step 14: Take a screenshot of the model summary. ------------------------------
 

## Step 15: Now, try out the model. Take a batch of  10  examples from the training data and call  model.predict   on it.
 

## Step 16: Provide a screenshot of the model summary. ------------------------------
 

## Step 17: Train the model.
 

## Step 18: Train the model for 1000 epochs, and record the training and validation accuracy in the  history  object.
EPOCHS = 1000    history = model.fit(    normed_train_data, train_labels,    epochs=EPOCHS, validation_split = 0.2, verbose=0,    callbacks=[tfdocs.modeling.EpochDots()])    
 

## Step 19: Visualize the model's training progress using the stats stored in the  history  object.
hist = pd.DataFrame(history.history)  hist['epoch'] = history.epoch  hist.tail()    
 

## Step 20: Provide a screenshot of the tail of the history. ------------------------------
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)    
plotter.plot({'Basic': history}, metric = "mae")  plt.ylim([0, 10])  plt.ylabel('MAE [MPG]')    
 

## Step 21: Provide a screenshot of the history plot. ------------------------------
plotter.plot({'Basic': history}, metric = "mse")  plt.ylim([0, 20])  plt.ylabel('MSE [MPG^2]')    
 

## Step 22: Compare the two models, one using Mean Absolute Error and the other using Mean Square Error.
Which fitted model is better? Are any other models “useful?”

## Step 23: For your deliverable, provide a detailed analysis using your screenshots as supporting content.
Write up your analysis using a Word document. Submit your Python code and Word document in a zip archive file. Name your archive file:

CSC580_CTA_3_2_last_name_first_name.zip.