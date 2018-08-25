from scipy.fftpack import fft
from scipy.signal import welch
import numpy as np
import tensorflow as tf
import scipy.io as spio
from scipy.signal import welch
import seaborn as sns; sns.set()

num_units = 50
signal_length = 2200 #time_steps
num_components = 2 #input_dim
num_labels = 2 #label_len

num_hidden = 32
learning_rate = 0.001
lambda_loss = 0.001
total_steps = 2000
display_step = 50
batch_size = 40

##################################################
sub_num = 24
time_steps = 200
over_lap = 0.5
type_input = 2
shuffle_flag = 1
global path

if (num_labels == 2):
    path= r'C:\Users\User\Documents\2017-2018\Project\MatlabScripts\load_data\mat_to_python_st.mat'

    TRAIN_INPUT = 'train_input_st'
    TRAIN_LABELS = 'train_labels_st'
    TEST_LABELS = 'test_labels_st'
    TEST_INPUT = 'test_input_st'

elif (num_labels == 6 or num_labels == 3):
    path = r'C:\Users\User\Documents\2017-2018\Project\MatlabScripts\load_data\mat_to_python.mat'

    TRAIN_INPUT = 'train_input'
    TRAIN_LABELS = 'train_labels'
    TEST_LABELS = 'test_labels'
    TEST_INPUT = 'test_input'


#Function which suffle labels and inputs.
def unison_shuffled_copies(a,b):
    '''
     array a and b  can have different shapes, but with the same length (leading dimension).
    :param a: array 1
    :param b: array 2
    :return: shuffle each of them, such that corresponding elements continue to correspond
    '''
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# type_input: 0 = ECG , 1 = GSR , 2 = both
def load_data_mat(path, shuffle_flag,type_input=2):
    """
    :param path: path of mat file with variables
    :param shuffle_flag: if shuffle_flag ==1 than shuffle all variables else no shuffle
    :return:
    """
    mat = spio.loadmat(path, squeeze_me=True)
    train_input =mat[TRAIN_INPUT]  # array
    train_labels = mat[TRAIN_LABELS] # structure containing an array
    test_labels = mat[TEST_LABELS]# array of structures
    test_input = mat[TEST_INPUT]

    # shuffle train and test
    if (shuffle_flag ==1):
        train_input,train_labels= unison_shuffled_copies(train_input,train_labels)
        test_input,test_labels= unison_shuffled_copies(test_input,test_labels)
    #input becomes a list of numpy 2d arrays, labels becomes a list of lists

    if (type_input!=2):
        train_input = [np.reshape(a[type_input,:],(time_steps,1)) for a in train_input]
        test_input = [np.reshape(a[type_input,:],(time_steps,1)) for a in test_input]
        train_labels = [a.tolist() for a in train_labels]
        test_labels = [a.tolist() for a in test_labels]
    else:
        train_input = [np.transpose(a) for a in train_input]
        test_input = [np.transpose(a) for a in test_input]
        train_labels = [a.tolist() for a in train_labels]
        test_labels = [a.tolist() for a in test_labels]

    train_input = np.array(train_input)
    test_input = np.array(test_input)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    return  train_input, test_input, train_labels, test_labels


train_dataset, test_dataset, train_labels, test_labels = load_data_mat(path, shuffle_flag,type_input)

print('subject:',sub_num)
print('labels:', num_labels)
print('type_input:', type_input)
print('over lap:' ,over_lap)
print('train_dataset shape:',  train_dataset.shape)
print('train_labels shape', train_labels.shape)


#############################################
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



if (type_input == 0): #ECG

    print('ECG: train_dataset.shape', train_dataset.shape)
    input_train_ecg = train_dataset[:,:,0]
    input_test_ecg = test_dataset[:,:,0]

    # Fit on training set only.
    scaler.fit(input_train_ecg)


    # Apply transform to both the training set and the test set.
    input_train_ecg = scaler.transform(input_train_ecg)
    print('ECG: train_dataset.shape', train_dataset.shape)
    print('train_dataset', train_dataset)
    input_test_ecg = scaler.transform(input_test_ecg)

    train_dataset[:,:,0] = input_train_ecg
    test_dataset[:,:,0] = input_test_ecg

elif(type_input == 1): #GSR

    print('GSR: train_dataset.shape', train_dataset.shape)
    input_train_gsr = train_dataset[:, :, 0]
    input_test_gsr = test_dataset[:, :, 0]

    # Fit on training set only.
    scaler.fit(input_train_gsr)

    # Apply transform to both the training set and the test set.
    input_train_ecg = scaler.transform(input_train_gsr)
    input_test_ecg = scaler.transform(input_test_gsr)

    train_dataset[:, :, 0] = input_train_gsr
    print('GSR: train_dataset', train_dataset)
    test_dataset[:, :, 0] = input_test_gsr
    print('GSR: test_dataset', test_dataset)

################################################


def accuracy(y_predicted, y):
    return (100.0 * np.sum(np.argmax(y_predicted, 1) == np.argmax(y, 1)) / y_predicted.shape[0])


###################################################################
#3.1 Building the model for a RNN
def rnn_model(data, num_hidden, num_labels):
    splitted_data = tf.unstack(data, axis=1)

    cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)
    #cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True) ##Added from 'network'
    outputs, current_state = tf.nn.static_rnn(cell, splitted_data, dtype=tf.float32)
    output = outputs[-1]

    w_softmax = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))
    b_softmax = tf.Variable(tf.random_normal([num_labels]))
    logit = tf.matmul(output, w_softmax) + b_softmax
    return logit
###################################################################
#3.2 From BasicRNNCell to BasicLSTMCell (and beyond)
#Since it does not have LSTM implemented, BasicRNNCell has its limitations.
# Instead of a BasicRNNCell we can use a BasicLSTMCell or an LSTMCell.
# Both are comparable, but a LSTMCell has some additional options like
# peephole structures, clipping of values, etc.
def rnn_lstm_model(data, num_hidden, num_labels):
    splitted_data = tf.unstack(data, axis=1)

    cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden)

    outputs, current_state = tf.nn.static_rnn(cell, splitted_data, dtype=tf.float32)
    output = outputs[-1]

    w_softmax = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))
    b_softmax = tf.Variable(tf.random_normal([num_labels]))
    logit = tf.matmul(output, w_softmax) + b_softmax
    return logit
###################################################################
#3.3 GruCell: A Gated Recurrent Unit Cell
#Besides BasicRNNCell and BasicLSTMCell,  Tensorflow also contains GruCell,
#which is an abstract implementation of the Gated Recurrent Unit,
#proposed in 2014 by Kyunghyun Cho et al.
def gru_rnn_model(data, num_hidden, num_labels):
    splitted_data = tf.unstack(data, axis=1)

    cell = tf.contrib.rnn.GRUCell(num_hidden)

    outputs, current_state = tf.nn.static_rnn(cell, splitted_data, dtype=tf.float32)
    output = outputs[-1]

    w_softmax = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))
    b_softmax = tf.Variable(tf.random_normal([num_labels]))
    logit = tf.matmul(output, w_softmax) + b_softmax
    return logit
###################################################################
#3.4 bi-directional LSTM RNN
def bidirectional_rnn_model(data, num_hidden, num_labels):
    splitted_data = tf.unstack(data, axis=1)

    lstm_cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
    outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_cell1, lstm_cell2, splitted_data, dtype=tf.float32)

    output = outputs[-1]

    w_softmax = tf.Variable(tf.truncated_normal([num_hidden * 2, num_labels]))
    b_softmax = tf.Variable(tf.random_normal([num_labels]))
    logit = tf.matmul(output, w_softmax) + b_softmax
    return logit
###################################################################
#3.5 Two-layered RNN
#We have seen how we can implement a bi-directional LSTM by stacking
#two LSTM Cells on top of each other, where the first on looks for sequential
# dependencies in the forward direction, and the second one
#in the backward direction. You could also place two LSTM cells
#on top of each other, simply to increase the neural network strength.
def twolayer_rnn_model(data, num_hidden, num_labels):
    splitted_data = tf.unstack(data, axis=1)

    cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
    cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2], state_is_tuple=True)

    outputs, state = tf.nn.static_rnn(cell, splitted_data, dtype=tf.float32)
    output = outputs[-1]

    w_softmax = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))
    b_softmax = tf.Variable(tf.random_normal([num_labels]))
    logit = tf.matmul(output, w_softmax) + b_softmax
    return logit
###################################################################
#3.6 Multi-layered RNN
#In this RNN network, n layers of RNN are stacked on top of each other.
#The output of each layer is mapped into the input of the next layer,
#and this allows the RNN to hierarchically looks for temporal dependencies.
#With each layer the representational power of the Neural Network increases (in theory).
def multi_rnn_model(data, num_hidden, num_labels, num_cells=4):
    splitted_data = tf.unstack(data, axis=1)

    lstm_cells = []
    for ii in range(0, num_cells):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True)
        lstm_cells.append(lstm_cell)

    cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells, state_is_tuple=True)
    outputs, state = tf.nn.static_rnn(cell, splitted_data, dtype=tf.float32)
    output = outputs[-1]

    w_softmax = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))
    b_softmax = tf.Variable(tf.random_normal([num_labels]))
    logit = tf.matmul(output, w_softmax) + b_softmax
    return logit
###################################################################

#data = tf.placeholder(tf.float32, [None, time_steps, input_dim]) #Maybe we want to change NONE to specific size
#target = tf.placeholder(tf.float32, [None, label_len])

graph = tf.Graph()
with graph.as_default():

    # 1) First we put the input data in a tensorflow friendly form.
    tf_dataset = tf.placeholder(tf.float32, shape=(None, signal_length, num_components))
    tf_labels = tf.placeholder(tf.float32, shape=(None, num_labels))

    # 2) Then we choose the model to calculate the logits (predicted labels)
    # We can choose from several models:
    #logits = rnn_model(tf_dataset, num_hidden, num_labels)##
    logits = rnn_lstm_model(tf_dataset, num_hidden, num_labels)
    # logits = bidirectional_lstm_rnn_model(tf_dataset, num_hidden, num_labels)
    #logits = twolayer_rnn_model(tf_dataset, num_hidden, num_labels)
    # logits = gru_rnn_model(tf_dataset, num_hidden, num_labels)
    #logits = multi_rnn_model(tf_dataset, num_hidden, num_labels, num_cells=4)

    # 3) Then we compute the softmax cross entropy between the logits and the (actual) labels
    l2 = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_labels)) + l2

    # 4.
    # The optimizer is used to calculate the gradients of the loss function
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss) # AdamOptimizer usually performs best.
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)

with tf.Session(graph=graph) as session:
    print("\nStarting to initialized")
    tf.global_variables_initializer().run()
    print("\nInitialized")
    for step in range(total_steps):
        # Since we are using stochastic gradient descent, we are selecting  small batches from the training dataset,
        # and training the convolutional neural network each time with a batch.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict = {tf_dataset: batch_data, tf_labels: batch_labels}
        _, l, train_predictions = session.run([optimizer, loss, prediction], feed_dict=feed_dict)
        train_accuracy = accuracy(train_predictions, batch_labels)

        if step % display_step == 0:
            feed_dict = {tf_dataset: test_dataset, tf_labels: test_labels}
            _, test_predictions = session.run([loss, prediction], feed_dict=feed_dict)
            test_accuracy = accuracy(test_predictions, test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {} %, accuracy on test set {:02.2f} %".format(
                step, l, train_accuracy, test_accuracy)
            print(message)







