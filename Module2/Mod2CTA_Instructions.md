Option #2: Predicting Future Sales
In this assignment, you will work with a neural network that can be used to predict future revenues from the sales of a new video game. A dataset is provided that you'll use to train a neural network to predict how much money you can expect future video games to earn based on historical data. The data are contained in a file named sales_data_training.csvLinks to an external site.. In this spreadsheet, there is one row for each video game that a store has sold in the past.

 

(Source: Geitgey, A. (n.d.). Building deep learnging: Keras Tree Master 03Links to an external site.. Github.)

The columns are defined as follows:

critic_rating : an average star rating out of five stars. 
is_action : tells us if this was an action game. 
is_exclusive_to_us : tells us if we have an exclusive deal to sell this game. 
is_portable : tells us if this game runs on a handheld video game system. 
is_role_ playing : tells us if this is a role-playing game, which is a genre of video game. 
is_sequel : tells us if this game was a sequel to an earlier video game and part of an ongoing series. 
is_sports : tell us if this was a sports game in the sports genre. 
suitable_for_kids tells us if this game is appropriate for all ages. 
total_earnings : tells us how much money the store has earned in total from selling the game to all customers.
unit_price tells us for how much a single copy of the game retailed. 
 

You’ll use Keras to train the neural network that will try to predict the total earnings of a new game based on these characteristics. Along with the sales_data_training.csv file, there is also a second data file called sales_data_test.csv. Links to an external site.This file is exactly like the first one. The machine learning system should only use the training dataset during the training phase. Then, you'll use the test data to check how well the neural network is working. To use this data to train a neural network, you first have to scale this data so that each value is between zero and one. Neural networks train best when data in each column is all scaled to the same range. Use the following Python code to scale the earning and unit price columns in both the training and test datasets. You will use Pandas to generate scaled training and test data sets.

 

import pandas as pdfrom sklearn.preprocessing import MinMaxScaler# Load training data set from CSV filetraining_data_df = pd.read_csv("sales_data_training.csv")# Load testing data set from CSV filetest_data_df = pd.read_csv("sales_data_test.csv")# Data needs to be scaled to a small range like 0 to 1 for the neural# network to work well.scaler = MinMaxScaler(feature_range=(0, 1))# Scale both the training inputs and outputsscaled_training = scaler.fit_transform(training_data_df)scaled_testing = scaler.transform(test_data_df)# Print out the adjustment that the scaler applied to the total_earnings column of dataprint("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[8], scaler.min_[8]))# Create new pandas DataFrame objects from the scaled datascaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)# Save scaled data dataframes to new CSV filesscaled_training_df.to_csv("sales_data_training_scaled.csv", index=False)scaled_testing_df.to_csv("sales_data_testing_scaled.csv", index=False)

 

Now that the data has been scaled, you’re ready to code the neural network. You’ll code a neural network with Keras. To do this, you’ll complete the given Python script named create_model.py, shown below.

 

import pandas as pd  from keras.models import Sequential  from keras.layers import *    training_data_df = pd.read_csv("sales_data_training_scaled.csv")     X = training_data_df.drop('total_earnings', axis=1).values  Y = training_data_df[['total_earnings']].values    # Define the model  model =
 
# Train the model
model.fit(…)
 
# Load the test data
# Load the separate test data set
test_data_df = pd.read_csv("sales_data_test_scaled.csv")
 
X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values
 
First, on line five, use the Python package, Pandas library, (visit the Python Pandas TutorialLinks to an external site. website for more information) to load the pre-scaled data from a CSV file. Each row of the dataset contains several features that describe each video game and then the total earnings value for that game. You want to split that data into two separate arrays: one with just the input features for each game and one with just the expected earnings. 

 

On line seven, to get just the input features, we grab all of the columns of the training data but drop the total earnings column. Then, on line eight, extract just the total earnings column as shown. Now, X contains all the input features for each game, and Y contains only the expected earnings for each game. Now, you can build a neural network starting on line 11. 

 

Incorporate the following parameters into your model definition:

use a sequential model
use nine inputs and one output
make the model dense
use the ReLU activation function for the hidden layers
use the linear activation function for the output layer.
 

Train your model using both X and Y as well as the following:

50 epochs
shuffle=True; this action will make Keras shuffle the data randomly during each epoch
verbose = 2; this tells Keras to print detailed information during the processing. Take a screenshot of these messages for your submission.
 

Evaluate your neural network model using model.evaluate(...) method. Print out the MSE for the test dataset.

 

test_error_rate = model.evaluate(…)  print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
 

Save your trained model. You will submit this model as part of your assignment.

# Save the model to disk  model.save("trained_model.h5")  print("Model saved to disk.") 
Next, you will load your trained model to make predictions. The prediction information is stored in proposed_new_product.csv and consists of one row. 

 

Complete the final segment of Python code. Be sure to rescale your final prediction using the two  parameters during the scaling of the training and testing data sets.

 

import pandas as pd  from keras.models import load_model    model = load_model('trained_model.h5')    X = pd.read_csv("proposed_new_product.csv").values  prediction = model.predict(…)    # Grab just the first element of the first prediction (since we only have one)  prediction = prediction[…][…]    # Re-scale the data from the 0-to-1 range back to dollars  # These constants are from when the data was originally scaled down to the 0-to-1 range  prediction = prediction + _____
prediction = prediction / _____    print("Earnings Prediction for Proposed Product - ${}".format(prediction))
 

For your deliverable, include an introduction of your finding in a Word Document. Submit your Python code and a screenshot showing the verbose run-time outputs, the testing MSE, and your final prediction in a zip archive file. Name your archive file:

