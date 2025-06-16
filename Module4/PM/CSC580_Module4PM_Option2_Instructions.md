# Option #2: Improving TensorFlow Model Performance and Quality
In this Portfolio Project Milestone, you will improve the results from the work started in Module 3’s Critical Thinking Assignment, Option 2. The derived models were not optimal and even present degradation after 100 epochs.

## Step 1
Update the model.fit call to automatically stop training when the validation score doesn't improve. Use an EarlyStopping callback that tests a training condition for every epoch. If a set amount of epochs elapses without showing improvement, then automatically stop the training. You can learn more about this callback hereLinks to an external site.

model = build_model()    # The patience parameter is the amount of epochs to check for improvement  

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)    

early_history = model.fit(normed_train_data, train_labels,
epochs=EPOCHS, validation_split = 0.2, verbose=0,
callbacks=[early_stop, tfdocs.modeling.EpochDots()])

plotter.plot({'Early Stopping': early_history}, metric = "mae")  
plt.ylim([0, 10])  
plt.ylabel('MAE [MPG]')      
 
## Step 2
Take a screenshot of the plot. What is the average error? Comment on the quality of this error.
 
## Step 3
Analyze how well the model generalizes by using the test set, which was not used when training the model. This tells us how well we can expect the model to predict when we use it in the real world.

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)    
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))    

## Step 4: Take a screenshot of the output.
 
## Step 5: Make predictions
Finally, predict MPG values using data in the testing set:

test_predictions = model.predict(normed_test_data).flatten()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]  plt.xlim(lims)
plt.ylim(lims)  
_ = plt.plot(lims, lims)    
         
Take a screenshot of the plot. Comment on the quality of the prediction.

## Step 6
Analyze the normality of the error distribution.

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

Take a screenshot of the plot and comment on the distribution.

# Deliverable
For your deliverable, provide a detailed analysis using your screenshots as supporting content. Write up your analysis using a Word document. Submit your Python code and Word document in a zip archive file. Name your archive file:

CSC580_MidTermPortfolio _Option_2_last_name_first_name.zip