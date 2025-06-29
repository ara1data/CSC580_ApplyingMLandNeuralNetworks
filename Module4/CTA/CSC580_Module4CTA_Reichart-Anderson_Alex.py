# Module 4 Critical Thinking Assignment
# Option 1: Toxicology Testing

# Alexander Reichart-Anderson
# MS in AI and ML, Colorado State University Global
# CSC580-1: Applying Machine Learning & Neural Networks - Capstone
# Dr. Joseph Issa
# June 8, 2025

# Step 0: Import Required Packages
import numpy as np
np.random.seed(456)
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.random.set_seed(456)
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score

# Step 1A: Load Tox21 Dataset
_, (train, valid, test), _ = dc.molnet.load_tox21()

# Step 1B: Extract features, labels, and weights from DeepChem datasets
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w

# Step 2: Remove extra datasets
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

# Step 3: Define placeholders that accept minibatches of different sizes
d = 1024
n_hidden = 50
learning_rate = 0.001
n_epochs = 10
batch_size = 100

with tf.name_scope("placeholders"):
    x = tf.compat.v1.placeholder(tf.float32, (None, d))
    y = tf.compat.v1.placeholder(tf.float32, (None,))
    # Step 6: Dropout placeholder
    keep_prob = tf.compat.v1.placeholder(tf.float32, ())

# Step 4 & 7: Hidden layer with dropout
with tf.name_scope("hidden-layer"):
    W = tf.Variable(tf.random.normal((d, n_hidden)))
    b = tf.Variable(tf.random.normal((n_hidden,)))
    x_hidden = tf.nn.relu(tf.matmul(x, W) + b)
    x_hidden = tf.nn.dropout(x_hidden, keep_prob)  # Apply dropout

# Step 5: Output layer
with tf.name_scope("output"):
    W = tf.Variable(tf.random.normal((n_hidden, 1)))
    b = tf.Variable(tf.random.normal((1,)))
    y_logit = tf.matmul(x_hidden, W) + b
    y_one_prob = tf.sigmoid(y_logit)
    y_pred = tf.round(y_one_prob)

with tf.name_scope("loss"):
    y_expand = tf.expand_dims(y, 1)
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
    l = tf.reduce_sum(entropy)

with tf.name_scope("optim"):
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(l)

with tf.name_scope("summaries"):
    tf.compat.v1.summary.scalar("loss", l)
    merged = tf.compat.v1.summary.merge_all()

# Step 8: Mini-batch training
train_writer = tf.compat.v1.summary.FileWriter('/tmp/fcnet-tox21', tf.compat.v1.get_default_graph())

N = train_X.shape[0]
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    step = 0
    
    for epoch in range(n_epochs):
        pos = 0
        while pos < N:  # Fixed loop condition
            batch_X = train_X[pos:pos+batch_size]
            batch_y = train_y[pos:pos+batch_size]
            
            feed_dict = {x: batch_X, y: batch_y, keep_prob: 0.5}  # 50% dropout during training
            _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
            
            print(f"epoch {epoch}, step {step}, loss: {loss}")
            train_writer.add_summary(summary, step)
            
            step += 1
            pos += batch_size
    
    # Validation with dropout disabled
    valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0})
    valid_acc = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
    print(f"Validation Accuracy: {valid_acc:.3f}")

# Step 9: Tensorboard Usage
# executed the following shell command to view the tensorboard graphs
# tensorboard --logdir=/tmp/fcnet-tox21
# go to http://localhost:6006/