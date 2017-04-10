#assignement 2 problem > to transfer to Jupyter Notebook

# Parameters
learning_rate = 0.05
num_step = 3001 #training cycle | training_epochs
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 1024 # 1st layer number of features
n_input = image_size * image_size # data input (img shape: 28*28)
#num_labels = 10 #(0-9)

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
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=pred))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(pred)
  valid_prediction = tf.nn.softmax(oneHiddenLayerModel(tf_valid_dataset, weights, biases))
  test_prediction = tf.nn.softmax(oneHiddenLayerModel(tf_test_dataset, weights, biases))

  ########
  with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
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
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))