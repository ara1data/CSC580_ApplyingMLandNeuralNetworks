# Module 3 Critical Thinking Assignment
# Option 2: Linear Regression Using TensorFlow

# Alexander Reichart-Anderson
# MS in AI and ML, Colorado State University Global
# CSC580-1: Applying Machine Learning & Neural Networks - Capstone
# Dr. Joseph Issa
# June 1, 2025

import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

# Enable TensorFlow 1.x compatibility
tf.disable_v2_behavior()

# Set reproducible random seeds
np.random.seed(101)
tf.set_random_seed(101)

# Generate random linear data with noise
x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)
n = len(x)

# Step 1: Plot training data
plt.scatter(x, y, label='Training Data')
plt.title("Random Linear Data with Noise")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Step 2: Create placeholders
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: Initialize trainable variables
W = tf.Variable(np.random.randn(), name='weight')
b = tf.Variable(np.random.randn(), name='bias')

# Step 4: Set hyperparameters
learning_rate = 0.01
training_epochs = 1000

# Step 5: Model components
hypothesis = tf.add(tf.multiply(W, X), b)
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize variables
init = tf.global_variables_initializer()

# Step 6: Training process
with tf.Session() as sess:
    sess.run(init)
    
    # Training loop
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={X: x, Y: y})
        
        # Display progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            c = sess.run(cost, feed_dict={X: x, Y: y})
            print(f"Epoch {epoch+1:4}: cost = {c:.4f}")
    
    # Step 7: Get final parameters
    training_cost = sess.run(cost, feed_dict={X: x, Y: y})
    weight = sess.run(W)
    bias = sess.run(b)
    
    print("\nTraining complete!")
    print(f"Final cost: {training_cost:.4f}")
    print(f"Weight: {weight:.4f}")
    print(f"Bias: {bias:.4f}")
    
    # Generate predictions for plotting
    predictions = weight * x + bias

# Step 8: Plot results
plt.scatter(x, y, label='Original Data')
plt.plot(x, predictions, 'r', label='Fitted Line')
plt.title("Linear Regression Results")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
