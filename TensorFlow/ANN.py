import tensorflow as tf # Native tensorflow library
import numpy as np #Numpy to be used to one-hot encoding the eval functions
from PIL import Image  #Pillow library to be used for images
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #use numpy's one-hot encoding to reformat labels
myWriter=tf.summary.FileWriter("H:\\MyImageTrainer\\") #Create the FileWriter to with the accuracy vs iterations graph and also the DFG
#====Define the layers which is 784 x 512 x 256 x 128 x 10 in this case=========#
n_input = 784   
n_hidden1 = 512 
n_hidden2 = 256 
n_hidden3 = 128 
n_output = 10   
#======Define learning rate, iterations, and batch size (to be used for calculating accuracy after each 100 steps)====#
learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5
#========Define the input and output layer==========#
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32) 
#=========Initial Weights==========#
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}
#========Biases of different layers========#
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}
#======Group the input operations of each layers in a name scope=======#
with tf.name_scope('Layer_1_Input'):
    layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])

with tf.name_scope('Layer_2_Input'):    
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])

with tf.name_scope('Layer_3_Input'):
    layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])

layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']
#Use the Tensorflow API calls to select the output based on comparisions of all neurons at output layer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)   

#Accuracy will be calculated by finding average of all prediction of mini batch of images
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()  #Initialize all placeholders
sess = tf.Session() #Create a session to perform operations
sess.run(init) #start session

print("\n\n=================Starting Traing============\n\n")
#========================================================================================#
for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})

    # write accuracy (per minibatch) after i iterations in summary to be plotted on Accuracy graph
    if i%100==0:
        summary=sess.run(merged, feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
        myWriter.add_summary(summary,i)
        print("Iteration", str(i)," Completed")
#========================================================================================#
print("\n\n===================Traing Completed===========\n\n")
        
test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1.0})
myWriter.add_graph(sess.graph)
print("\n\nAccuracy on test set:", test_accuracy)

#=======================Testing some images==============================================#

#For image 2
img = np.invert(Image.open("H:\\MyImageTrainer\\test_img.png").convert('L')).ravel()
prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [img]})
print ("\nPrediction for test image:", np.squeeze(prediction))

#For image containig 3
img = np.invert(Image.open("H:\\MyImageTrainer\\three.png").convert('L')).ravel()
prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [img]})
print ("\nPrediction for test image:", np.squeeze(prediction))

#For image containig 6
img = np.invert(Image.open("H:\\MyImageTrainer\\six.png").convert('L')).ravel()
prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [img]})
print ("\nPrediction for test image:", np.squeeze(prediction))