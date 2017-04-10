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