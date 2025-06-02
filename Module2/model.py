import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Load scaled training data
training_data_df = pd.read_csv("Module2/outputdata/sales_data_training_scaled.csv")
X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

# Define model architecture
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu'))  # First hidden layer
model.add(Dense(50, activation='relu'))               # Second hidden layer
model.add(Dense(1, activation='linear'))              # Output layer

# Compile and train
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=50, shuffle=True, verbose=2)

# Evaluate on test data
test_data_df = pd.read_csv("Module2/outputdata/sales_data_training_scaled.csv")
X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values
test_mse = model.evaluate(X_test, Y_test)
print(f"Test MSE: {test_mse}")

# Save model
model.save("trained_model.h5")
