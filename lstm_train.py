from data import VCTK
import tensorflow as tf

data = VCTK([225, 226, 227])

sequence = tf.placeholder(tf.float32, shape=(20, 500))

n_speakers = 3
n_inputs = 20 * 500 # Pseudo-images are 20 * 500
learning_rate = 0.00001
batch_size = 60
training_iters = 100000
dropout = 0.75
display_step = 100

with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, [None, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_speakers])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Sequence Patch:
# ^ MFCC feature (20)
# |
# |
# -------> logic time axis (500)
#
# If the sequence are more than 500 time unit
# it will be truncated at the 500th time unit.
#
# If the sequence is less than 500 time unit 
# if will be padded out with zeroes

"""Helper for creating a conv2d"""
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

"""Helper for creating a maxpool2d"""
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases, dropout):
    # The input is just x reshapes to the right format, 
    # -1 is for the uknown size of the batch
    x = tf.reshape(x, [-1, 20, 500, 1])

    # Convolution layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max pooling (or down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# zero-padding the 20x500x1 image to 24x504x1
# applying 5x5x32 conv to get 20x500x32
# max-pooling to 10x250x32
# zero padding 10x250x32 to 14x254x32
# applying 5x5x32x64 conv to get 10x250x64
# max-pooling to get 5x125x64

# Store the weights
with tf.name_scope('Weights'):
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5,5,1,32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # Fully connected layer 5*125*64 input (see above), 1024 output
        'wd1': tf.Variable(tf.random_normal([5*125*64, 1024])),
        # 1024 input, 2 outputs 
        'out': tf.Variable(tf.random_normal([1024, n_speakers]))
    }

# Store the biases
with tf.name_scope('Biases'):
    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_speakers]))
    }

# Get the model
prediction = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
with tf.name_scope('Cross-entropy'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

with tf.name_scope('Train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Write summary
tf.scalar_summary("cost", cost)
tf.scalar_summary("accuracy", accuracy)

summary_op = tf.merge_all_summaries()

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1

    # Log writer
    writer = tf.train.SummaryWriter('data/train/log', graph=tf.get_default_graph())

    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = data.next_batch(batch_size)
        # Run optimization op (backprop)
        _, summary = sess.run([optimizer, summary_op], feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        # Write log
        writer.add_summary(summary, step)

        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    """print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: data.features,
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))"""