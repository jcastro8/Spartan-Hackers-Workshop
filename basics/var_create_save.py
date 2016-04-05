#Source File: var_create_save.py
#Author: Juan L. Castro-Garcia
#Date: April 5, 2016
#Description: Show how to create, initialize, save and load variables

# Use the TensorFLow library
import tensorflow as tf

# Create variables Wx + b formula

x = tf.Variable(tf.random_normal([2,5], mean=1.0, stddev = 0.5), name="x")

weights = tf.Variable(tf.random_normal([5,2], stddev = 0.35), name="weights")

biases = tf.Variable(tf.zeros([2]), name="biases")

y = tf.matmul(x,weights) + biases


# Initialize a variable with the values of another
w2 = tf.Variable(weights.initialized_value(), name="w2")

y2 = tf.matmul(x,w2) + biases


# To save the variable, we need to create a Saver op ( or object for simplicity)
saver = tf.train.Saver()


# Initialize the variables
init_op = tf.initialize_all_variables()

# Open the session
session = tf.Session()

# Run the session
session.run(init_op)

# save the model
s_path = "./model.ckpt"
save_path = saver.save(session, s_path)
print("Model saved in file: %s" % save_path)

# restore the model
saver.restore(session,s_path)
print("Model is restored")

print("the result from y = (weights * x) + biases") 
print(session.run(y))

print("the result from y2 = (w2 * x) + biases") 
print(session.run(y))

