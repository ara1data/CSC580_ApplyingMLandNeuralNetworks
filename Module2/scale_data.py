import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load training and test data
training_data_df = pd.read_csv("Module2/datasets/sales_data_training.csv")
test_data_df = pd.read_csv("Module2/datasets/sales_data_test.csv")

# Initialize scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale datasets
scaled_training = scaler.fit_transform(training_data_df)
scaled_testing = scaler.transform(test_data_df)

# Save scaled data
pd.DataFrame(scaled_training, columns=training_data_df.columns.values).to_csv("Module2/outputdata/sales_data_training_scaled.csv", index=False)
pd.DataFrame(scaled_testing, columns=test_data_df.columns.values).to_csv("Module2/outputdata/sales_data_testing_scaled.csv", index=False)

# Print scaling parameters for total_earnings (column index 8)
print(f"Note: total_earnings scaled by multiplying {scaler.scale_[8]:.10f} and adding {scaler.min_[8]:.6f}")
