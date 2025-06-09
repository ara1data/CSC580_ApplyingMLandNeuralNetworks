Option #1: Linear Regression Using TensorFlow
In this assignment, you will use TensorFlow to predict the next output from a given set of random inputs. Start by importing the necessary libraries. You will use Numpy along with TensorFlow for computations and Matplotlib for plotting.

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

#In order to make the random numbers predictable, we will define fixed seeds for both Numpy and #TensorFlow.

np.random.seed(101)

tf.set_random_seed(101)

#Now, let’s generate some random data for training the Linear Regression Model.

# Generating random linear data

# There will be 50 data points ranging from 0 to 50

x = np.linspace(0, 50, 50)

y = np.linspace(0, 50, 50)

# Adding noise to the random linear data

x += np.random.uniform(-4, 4, 50)

y += np.random.uniform(-4, 4, 50)

n = len(x) # Number of data points

Complete the following steps:

1) Plot the training data.

2) Create a TensorFlow model by defining the placeholders X and Y so that you can feed your training examples X and Y into the optimizer during the training process.

3) Declare two trainable TensorFlow variables for the weights and bias and initialize them randomly.

4) Define the hyperparameters for the model:

     learning_rate = 0.01

     training_epochs = 1000

5) Implement Python code for:

the hypothesis,
the cost function,
the optimizer.
6) Implement the training process inside a TensorFlow session.

7) Print out the results for the training cost, weight, and bias.

8) Plot the fitted line on top of the original data.

For your deliverable, submit an introduction in a Word document. Submit your Python code and screenshots of your plots in a zip archive file. Name your archive file:

CSC580_CTA_3_1_last_name_first_name.zip.