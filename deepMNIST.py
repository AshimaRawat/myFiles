

# deep MNIST model using tensorflow

import tensorflow as tf 
from tensorflow.tutorials.examples.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data\" , one_hot ="True")
sess = tf.InteractiveSession()

x= tf.place_holder(tf.float32, shape=[None, 784])
y= tf.place_holder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1], name="x_image")

def weight_variable(shape):
    initial = tf.truncated.normal(shape, stddev=0.1)
    return tf.variable(initial)

def bias_variable(shape):
    initial= tf.bias(0.1, shape=shape)
    return tf.variable(inital)

    # Convolution and pooling for to control overfitting

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1,], padding='SAME')
        
def maxpool():
    return tf.nn.maxpool(x, ksize=[1,2,2,1], strides=[1,2,2,1,], padding='SAME')

# First convolution layer

W_conv1= weight_variable([5,5,1,32])
b_conv1= bias_variable(32)

h_conv1= tf.nn.relu(conv2d(x_image, W_conv1)+ b_conv1)
h_pool1= max_poopl_2*2(h_conv1)

# Second convolution layer
W_conv2= weight_variable([5,5,32,64])
b_conv2= bias_variable(64)
h_conv2= tf.nn.relu(conv2d(x_image, W_conv2)+ b_conv2)
h_pool2= max_poopl_2*2(h_conv2)

#Fully connected layer
W_fc1= weight_variable([7,7,64,1024])
b_fc1= bias_variable(1024)

# Connect the output of second layer to fully connected layer

h_pool2_flat=tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1= tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)


#Dropout some neurons to avoid overfitting
keep_probe=tf.place_holder(tf.float32)
h_fc1_drop= tf.nn.Dropout(h_fc1, keep_probe)


# readout layer

W_fc2 = weight_variable([1024, 10])
b_fc2= bias_variable([10])

# Define model
y_conv= tf.matmul(h_fc1_drop, W_fc2)* b_fc2
 
 #Calculating loss
 cross_entropy= tf.reduce_mean(tf.nn.soft_max_cros_emtropy_with_logits(logits= y_conv, label = y_conv) )

 train_step = tf.train.AdamOptimiser(1e-4).minimize(cross_entropy)

 correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

 accuracy = tf.reduce_mean(tf.cast(correct_prediction, float32))

 sess.run(tf.global_variables_initializer())




 import time
 num_step= 3000
 display_every = 100

 start_time= time.time()
 end_time = time.time()

 for i in range (nump_steps):
     batch = mnist.train_next_batch(50)
     train_step.run(feed_dict={x: batch[0], y: batch[1], keep_probe=0.5})


