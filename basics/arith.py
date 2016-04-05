#Source File: arith.py
#Author: Juan L. Castro-Garcia
#Date: April 5, 2016
#Description: Simple arithmetic operations

import tensorflow as tf

# placeholders are symbolic variables (not initialized/created)
# must feed values before running

a = tf.placeholder("float")
b = tf.placeholder("float")

# values for operations
v1 = 4
v2 = 5

mult = tf.mul(a,b)
add = tf.add(a,b)
sub = tf.sub(a,b)
div = tf.div(a,b)

# Create the session
sess = tf.Session()

print "%d + %d = %f" % (v1,v2,sess.run(add, feed_dict={a: v1, b: v2}))
print "%d - %d = %f" % (v1,v2,sess.run(sub, feed_dict={a: v1, b: v2}))
print "%d * %d = %f" % (v1,v2,sess.run(mult, feed_dict={a: v1, b: v2}))
print "%d / %d = %f" % (v1,v2,sess.run(div, feed_dict={a: v1, b: v2}))
