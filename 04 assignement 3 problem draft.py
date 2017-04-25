#assignement 3 : regularization techniques

"""
Probleme 1
Introduce and tune L2 regularization for both logistic and neural network models. Remember that L2 amounts to adding a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor t using nn.l2_loss(t). The right amount of regularization should improve your validation / test accuracy.
"""
# With logistic regression
train_subset = 10000
reg_term = 0.01

graph = tf.Graph()
with graph.as_default():

  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
  tf_train_labels = tf.constant(train_labels[:train_subset])
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
  weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  unreg_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  l2_loss = reg_term * tf.nn.l2_loss(weights)
  loss = tf.reduce_mean(unreg_loss + l2_loss)
  
  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)



num_steps = 801

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(
        predictions, train_labels[:train_subset, :]))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


########################################################################
# with one hidden layer network
# Parameters
learning_rate = 0.05
num_steps = 30001 #training cycle | training_epochs
batch_size = 512
n_hidden_1 = 1024 # 1st layer number of features
n_input = image_size * image_size # data input (img shape: 28*28)
#num_labels = 10 #(0-9)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def oneHiddenLayerModel (x, weights, biases):
  # Hidden layer with RELU activation
  hiddenLayer = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
  hiddenLayer = tf.nn.relu(hiddenLayer)
  # Output layer with linear activation
  outputLayer = tf.add(tf.matmul(hiddenLayer, weights['out']), biases['out'])
  return outputLayer

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, n_input))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  reg_term = tf.placeholder(tf.float32) # regularization term (beta)
  
  # Variables : layers weight & bias
  weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_1, num_labels]))
  }
  biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'out': tf.Variable(tf.zeros([num_labels]))
  }
 
  # Construct model
  pred = oneHiddenLayerModel(tf_train_dataset, weights, biases)

  # Loss (Cost)
  unreg_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf_train_labels))
  l2_loss = reg_term * (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['out']))
  loss = tf.reduce_mean(unreg_loss + l2_loss)
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(pred)
  valid_prediction = tf.nn.softmax(oneHiddenLayerModel(tf_valid_dataset, weights, biases))
  test_prediction = tf.nn.softmax(oneHiddenLayerModel(tf_test_dataset, weights, biases))

########################################################################
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print ('Initialized')
  summary_data = np.zeros((num_steps, 3))

  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, reg_term : 1e-2}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

    if (step % 5000 == 0):
	  valid_accuracy = accuracy(valid_prediction.eval(), valid_labels)
      summary_data[step] = [step, valid_accuracy, l]
      print ("Minibatch loss at step", step, ":", l)
      print ("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print ("Validation accuracy: %.1f%%" % valid_accuracy)
  print ("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))



#### plotting https://github.com/rndbrtrnd/udacity-deep-learning/blob/master/3_regularization.ipynb

########################################################################

#Problem 2 extreme overfitting case
num_bacthes = 3

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print ('Initialized')
  summary_data = np.zeros((num_steps, 3))

  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    ##offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    offset = step % num_bacthes
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, reg_term : 1e-2}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

    if (step % 5000 == 0):
	  valid_accuracy = accuracy(valid_prediction.eval(), valid_labels)
      summary_data[step] = [step, valid_accuracy, l]
      print ("Minibatch loss at step", step, ":", l)
      print ("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print ("Validation accuracy: %.1f%%" % valid_accuracy)
  print ("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

########################################################################

"""
Problem 3 
Introduce Dropout on the hidden layer of the neural network. 
Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. 
TensorFlow provides nn.dropout() for that, but you have to make sure it's only inserted during training.
What happens to our extreme overfitting case?
"""
# with one hidden layer network
# Parameters
learning_rate = 0.05
num_steps = 30001 #training cycle | training_epochs
batch_size = 512
dropout = 0.5
n_hidden_1 = 1024 # 1st layer number of features
n_input = image_size * image_size # data input (img shape: 28*28)
#num_labels = 10 #(0-9)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def oneHiddenLayerModel (x, weights, biases, dropout):
  # Hidden layer with RELU activation
  hiddenLayer = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
  hiddenLayer = tf.nn.relu(hiddenLayer)
  # Apply dropout
  hiddenLayer = tf.nn.dropout(hiddenLayer, dropout)
  # Output layer with linear activation
  outputLayer = tf.add(tf.matmul(hiddenLayer, weights['out']), biases['out'])
  return outputLayer

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, n_input))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  reg_term = tf.placeholder(tf.float32) # regularization term (beta)
  
  # Variables : layers weight & bias
  weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_1, num_labels]))
  }
  biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'out': tf.Variable(tf.zeros([num_labels]))
  }
 
  # Construct model
  pred = oneHiddenLayerModel(tf_train_dataset, weights, biases, keep_prob)

  # Loss (Cost)
  unreg_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf_train_labels))
  l2_loss = reg_term * (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['out']))
  loss = tf.reduce_mean(unreg_loss + l2_loss)
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(pred)
  valid_prediction = tf.nn.softmax(oneHiddenLayerModel(tf_valid_dataset, weights, biases, 1)) #TO CHECK!!
  test_prediction = tf.nn.softmax(oneHiddenLayerModel(tf_test_dataset, weights, biases, 1)) #TO CHECK!!

########################################################################
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print ('Initialized')
  summary_data = np.zeros((num_steps, 3))

  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, reg_term : 1e-2, keep_prob: dropout}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

    if (step % 5000 == 0):
	  valid_accuracy = accuracy(valid_prediction.eval(), valid_labels)
      summary_data[step] = [step, valid_accuracy, l]
      print ("Minibatch loss at step", step, ":", l)
      print ("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print ("Validation accuracy: %.1f%%" % valid_accuracy)
  print ("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


################################################################################
"""
Problem 4
Try to get the best performance you can using a multi-layer model! The best reported test accuracy using a deep network is 97.1%.
One avenue you can explore is to add multiple layers.
Another one is to use learning rate decay:
	global_step = tf.Variable(0)  # count the number of steps taken.
	learning_rate = tf.train.exponential_decay(0.5, global_step, ...)	
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
"""



#################################################################################################
#https://www.kaggle.com/segovia/digit-recognizer/tensorflow-nn-without-convolution
# coding: utf-8
'''
# Neural Nets with 3 hidden layers (size 1024, 512, 128)
# <li>1. l2 regularization and relu.</li> 
# <li>2. Use exponetial decay learning rate for GradientDescentOptimizer</li>
# <li>3. Use small batch to traing network at each step.</li>
# 
# These codes are adapted from an assignment I finished during taking the <a href='https://www.udacity.com/course/deep-learning--ud730'>Udacity Deep Learning Course</a>.
# 
# The nudge_dataset function is adapted from <a href='http://scikit-learn.org/stable/auto_examples/neural_networks/plot_rbm_logistic_classification.html#example-neural-networks-plot-rbm-logistic-classification-py
# ' >an example</a> found on scikit-learn.org.
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from scipy.ndimage import convolve

# read in data
train = pd.read_csv('../input/train.csv', dtype='float32')

# X_real_test is the 28000 images need to be classified/predicted.
X_real_test = pd.read_csv('../input/test.csv', dtype='float32')


# X: features
X = train.ix[:,1:785]
# scale data proportionally
XS = X / 255
XS_real_test = X_real_test / 255


# Y: labels
Y = train.ix[:,0]
Y2 = np.zeros([42000, 10])

# convert the Y series to a matrix with 0 or 1 to indicate the digits
for i in range(42000):
    #print ("i = %f" % i)
    index = Y[i]
    Y2[i, index] = 1


# split into train and a second part, 70% and 30%
from sklearn.cross_validation import train_test_split
XS_train, XS_second, Y_train, Y_second = train_test_split(XS, 
                                                          Y2, 
                                                          test_size = 0.3, 
                                                          random_state = 0)


# split the second part into validation and test, 50% and 50%
XS_valid, XS_test, Y_valid, Y_test = train_test_split(XS_second, 
                                                      Y_second, 
                                                      test_size = 0.5, 
                                                      random_state = 0)


# convert dataframes into ndarrays
Xvalidm = XS_valid.as_matrix()
Xtestm = XS_test.as_matrix()
XrealTestm = XS_real_test.as_matrix()


"""
This produces a dataset 5 times bigger than the original one,
by moving the 28x28 images in X around by 1px to left, right, down, up
"""
def nudge_dataset(X,Y):

    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((28, 28)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y
    


# nudge to get 5X training set
Xn, Yn = nudge_dataset(XS_train, Y_train)

# nudge Xn and Yn from last step to get 25X training set
Xtrainm, Ytrain = nudge_dataset(Xn, Yn)


def accuracy(predictions, labels):
    correct_pred_num = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    return (100.0 * correct_pred_num / predictions.shape[0])


# define graph

num_labels = 10

batch_size = 128

hidden_layer1_size = 1024
hidden_layer2_size = 512
hidden_layer3_size = 128

hidden_layer1_stdev = np.sqrt(2.0 / 784)
hidden_layer2_stdev = np.sqrt(2.0 / hidden_layer1_size)
hidden_layer3_stdev = np.sqrt(2.0 / hidden_layer2_size)
output_layer_stdev = np.sqrt(2.0 / hidden_layer3_size)

hidden_layer1_keepprob = 0.5
hidden_layer2_keepprob = 0.7
hidden_layer3_keepprob = 0.8

beta1 = 1e-5
beta2 = 1e-5
beta3 = 1e-5
beta4 = 1e-5

deep_graph = tf.Graph()
with deep_graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, 784))
    tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_valid_dataset = tf.constant(Xvalidm)
    tf_test_dataset = tf.constant(Xtestm)
    tf_real_test_dataset = tf.constant(XrealTestm)
    
    # first hidden layer
    hidden_layer1_weights = tf.Variable(
        tf.truncated_normal([784, hidden_layer1_size], stddev = hidden_layer1_stdev))
    hidden_layer1_biases = tf.Variable(tf.zeros([hidden_layer1_size]))
    hidden_layer1 = tf.nn.dropout(
        tf.nn.relu(tf.matmul(tf_train_dataset, hidden_layer1_weights) + hidden_layer1_biases),
        hidden_layer1_keepprob)
    
    # second hidden layer
    hidden_layer2_weights = tf.Variable(
        tf.truncated_normal([hidden_layer1_size, hidden_layer2_size], stddev = hidden_layer2_stdev))
    hidden_layer2_biases = tf.Variable(tf.zeros([hidden_layer2_size]))
    hidden_layer2 = tf.nn.dropout(
        tf.nn.relu(tf.matmul(hidden_layer1, hidden_layer2_weights) + hidden_layer2_biases),
        hidden_layer2_keepprob)
    
    # third hidden layer
    hidden_layer3_weights = tf.Variable(
        tf.truncated_normal([hidden_layer2_size, hidden_layer3_size], stddev = hidden_layer3_stdev))
    hidden_layer3_biases = tf.Variable(tf.zeros([hidden_layer3_size]))
    hidden_layer3 = tf.nn.dropout(
        tf.nn.relu(tf.matmul(hidden_layer2, hidden_layer3_weights) + hidden_layer3_biases),
        hidden_layer3_keepprob)
    
    # output layer
    output_weights = tf.Variable(
        tf.truncated_normal([hidden_layer3_size, num_labels], stddev = output_layer_stdev))
    output_biases = tf.Variable(tf.zeros([num_labels]))
    logits = tf.matmul(hidden_layer3, output_weights) + output_biases
    
    # calculate the loss with regularization
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    loss += (beta1 * tf.nn.l2_loss(hidden_layer1_weights) +
             beta2 * tf.nn.l2_loss(hidden_layer2_weights) +
             beta3 * tf.nn.l2_loss(hidden_layer3_weights) + 
             beta4 * tf.nn.l2_loss(output_weights))
    
    # learn with exponential rate decay
    global_step = tf.Variable(0, trainable = False)
    starter_learning_rate = 0.4
    learning_rate = tf.train.exponential_decay(starter_learning_rate, 
                                               global_step, 
                                               100000,
                                               0.96,
                                               staircase = True)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
    train_prediction = tf.nn.softmax(logits)
    
    ## setup validation prediction
    valid_hidden_layer1 = tf.nn.relu(tf.matmul(tf_valid_dataset, 
                                               hidden_layer1_weights) + hidden_layer1_biases)
    valid_hidden_layer2 = tf.nn.relu(tf.matmul(valid_hidden_layer1, 
                                               hidden_layer2_weights) + hidden_layer2_biases)
    valid_hidden_layer3 = tf.nn.relu(tf.matmul(valid_hidden_layer2,
                                               hidden_layer3_weights) + hidden_layer3_biases)
    valid_logits = tf.matmul(valid_hidden_layer3, output_weights) + output_biases
    valid_prediction = tf.nn.softmax(valid_logits)
    
    ## set up test prediction
    test_hidden_layer1 = tf.nn.relu(tf.matmul(tf_test_dataset,
                                              hidden_layer1_weights) + hidden_layer1_biases)
    test_hidden_layer2 = tf.nn.relu(tf.matmul(test_hidden_layer1,
                                              hidden_layer2_weights) + hidden_layer2_biases)
    test_hidden_layer3 = tf.nn.relu(tf.matmul(test_hidden_layer2,
                                              hidden_layer3_weights) + hidden_layer3_biases)
    test_logits = tf.matmul(test_hidden_layer3, output_weights) + output_biases
    test_prediction = tf.nn.softmax(test_logits)
    
    ## set up real_test_prediction
    real_test_layer1 = tf.nn.relu(tf.matmul(tf_real_test_dataset,
                                            hidden_layer1_weights) + hidden_layer1_biases)
    real_test_layer2 = tf.nn.relu(tf.matmul(real_test_layer1,
                                            hidden_layer2_weights) + hidden_layer2_biases)
    real_test_layer3 = tf.nn.relu(tf.matmul(real_test_layer2,
                                            hidden_layer3_weights) + hidden_layer3_biases)
    real_test_logits = tf.matmul(real_test_layer3, output_weights) + output_biases
    real_test_prediction = tf.nn.softmax(real_test_logits)


# run session

num_steps = 15001

train_acc_records = np.zeros(num_steps)
valid_acc_records = np.zeros(num_steps)
test_acc_records = np.zeros(num_steps)
loss_records = np.zeros(num_steps)

start_time = time.time()

with tf.Session(graph=deep_graph) as sess:
    tf.initialize_all_variables().run()
    print ("Initialized")
    
    for step in range(num_steps):
        offset = (step * batch_size) % (Ytrain.shape[0] - batch_size)
        
        batch_data = Xtrainm[offset:(offset + batch_size),:]
        batch_labels = Ytrain[offset:(offset + batch_size),:]
        
        feed_dict = {tf_train_dataset:batch_data,
                     tf_train_labels: batch_labels}
        
        _,l,pred = sess.run(
         [optimizer, loss, train_prediction], feed_dict = feed_dict)
        
        train_acc_records[step] = accuracy(pred, batch_labels)
        valid_acc_records[step] = accuracy(valid_prediction.eval(), Y_valid)
        test_acc_records[step] = accuracy(test_prediction.eval(), Y_test)
        loss_records[step] = l
        
        if (step % 1000) == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %0.1f%%" % accuracy(pred, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), Y_valid))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), Y_test))
    
    print(" ")
    print("Now let's make a prediction!")
    real_prediction_results = real_test_prediction.eval()
    print ("Done!")
    time_elapsed = time.time() - start_time
    print ("Run time is approx. %s minutes" % str(int(time_elapsed/60)))

