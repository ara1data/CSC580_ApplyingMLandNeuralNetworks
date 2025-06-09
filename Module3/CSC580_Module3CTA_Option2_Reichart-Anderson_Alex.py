# Module 3 Critical Thinking Assignment
# Option 2: Predicting Fuel Efficiency Using TensorFlow 

# Alexander Reichart-Anderson
# MS in AI and ML, Colorado State University Global
# CSC580-1: Applying Machine Learning & Neural Networks - Capstone
# Dr. Joseph Issa
# June 1, 2025

# ------------------------------
# To Install OpenCV: pip install face_recognition
# ------------------------------

# Requirement 0: imports
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Data Preparation ---
# Step 1: Download and load dataset
dataset_path = keras.utils.get_file(
    "auto-mpg.data", 
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)

# Step 2: Import and clean data (same as before)
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(
    dataset_path,
    names=column_names,
    na_values="?",
    comment='\t',
    sep=" ",
    skipinitialspace=True
)
dataset = raw_dataset.dropna()
dataset['Origin'] = dataset['Origin'].astype('category').cat.codes

# Step 3: Split data FIRST
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Step 4: Remove target column EARLY
train_labels = train_dataset.pop('MPG')  # Now 7 features remain
test_labels = test_dataset.pop('MPG')    # 7 features here too

# Step 5: Calculate statistics AFTER removing target
train_stats = train_dataset.describe().transpose()
train_stats.to_csv('training_statistics.csv')

# Step 6: Normalization function
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

# Step 7: Apply normalization
normed_train_data = norm(train_dataset)  # Correct shape: (None, 7)
normed_test_data = norm(test_dataset)    # Correct shape: (None, 7)

# --- MODEL BUILDING (remainder unchanged) ---
# Step 8: Build model with CORRECT input shape
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),  # Now 7
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model

# Step 10: Separate labels
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Step 11: Normalize data
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# --- Model Construction ---
# Step 12: Build model
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse']
    )
    return model

model = build_model()

# Step 13: Save model summary
with open('model_architecture.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# --- Model Training ---
# Step 17: Train with early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

EPOCHS = 1000
history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[
        early_stop,
        tfdocs.modeling.EpochDots()
    ]
)

# Save training history
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.to_csv('training_history.csv')

# --- Visualization ---
# Generate MAE plot
mae_plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
mae_plotter.plot({'Training History': history}, metric='mae')
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.savefig('mae_progress.png', bbox_inches='tight')
plt.close()

# Generate MSE plot
mse_plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
mse_plotter.plot({'Training History': history}, metric='mse')
plt.ylim([0, 20])
plt.ylabel('MSE [MPG²]')
plt.savefig('mse_progress.png', bbox_inches='tight')
plt.close()

# --- Model Evaluation ---
# Generate predictions
test_predictions = model.predict(normed_test_data).flatten()

# Calculate evaluation metrics
mae = tf.keras.metrics.mean_absolute_error(test_labels, test_predictions).numpy()
mse = tf.keras.metrics.mean_squared_error(test_labels, test_predictions).numpy()

# Save evaluation results
with open('model_performance.txt', 'w') as f:
    f.write(f'Test MAE: {mae:.2f} MPG\n')
    f.write(f'Test MSE: {mse:.2f} MPG²\n')

# Save sample predictions
pd.DataFrame({
    'Actual MPG': test_labels,
    'Predicted MPG': test_predictions,
    'Absolute Error': np.abs(test_labels - test_predictions)
}).head(10).to_csv('sample_predictions.csv')

print(f'Training complete. Final metrics:\nMAE: {mae:.2f}, MSE: {mse:.2f}')