##nnscript 
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import pickle

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1.0 / (1.0 + np.exp(-z))# your code here

selected_features = None
def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]
    global selected_features

    # Feature selection
    # Your code here.
    #Find columns in `train_data` where all values are the same
    var_features = np.all(train_data == train_data[0, :], axis=0)
    # Remove these constant features columns from train, validation, and test dataset
    train_data = train_data[:, ~var_features]
    validation_data = validation_data[:, ~var_features]
    test_data = test_data[:, ~var_features]

    # Storing the selected feature indices
    selected_features = np.where(~var_features)[0]

    print('Preprocess done')
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, the training data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of nodes in input layer (not including the bias node)
    % n_hidden: number of nodes in hidden layer (not including the bias node)
    % n_class: number of nodes in output layer (number of classes in
    %     classification problem)
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: vector of true labels of training images. Each entry
    %     in the vector represents the true label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing the value of the error function
    % obj_grad: a SINGLE vector of gradient values of the error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of the error function
    % for each weight in the weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weights w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layer.
    %     w1(i, j) represents the weight of the connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layer.
    %     w2(i, j) represents the weight of the connection from unit j in hidden 
    %     layer to unit i in output layer."""

    # Unpack arguments
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    # Reshape `params` vector into w1 and w2 matrices
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    # Initialize objective function value
    obj_val = 0

    # Number of samples
    n_train = training_data.shape[0]
    # Add bias to the input training data
    training_data_bias = np.hstack((training_data, np.ones((n_train, 1))))

    # Forward Pass
    # Calculates the hidden layer outputs by applying sigmoid function
    #z = sigmoid(np.dot(training_data_bias, w1.T))
    zj = sigmoid(np.dot(training_data_bias, np.transpose(w1)))

    # Adding bias term to hidden layer output
    #z_bias = np.hstack((z, np.ones((z.shape[0], 1))))
    zj_bias = np.concatenate((zj, np.ones((zj.shape[0], 1))), axis=1)
    # Calculates the final output o1 by applying sigmoid function to bais and weights w2
    ol = sigmoid(np.dot(zj_bias, np.transpose(w2)))


    # Converting labels 
    # Initializing the values to be zeros
    yl = np.zeros((n_train, n_class)) 
    # Setting the labels to 1 
    t = training_label.astype(int)
    yl[np.arange(n_train), t] = 1

    

    # Error function and Regularised error function 
    # Calculating the loss-function for our classification tasks 
    error = -np.sum(yl * np.log(ol) + (1 - yl) * np.log(1 - ol)) / n_train
    
    # Regularization error
    rgln_term = (lambdaval / (2 * n_train))
    #Calculating regularization error to prevent overfitting by penalizing large weights in the model.
    rgln_err =  rgln_term * (np.sum(w1 ** 2) + np.sum(w2 ** 2))
    # Total objective function value
    obj_val = error + rgln_err


    # Backpropagation
    # Calculates the Output layer error
    delta_output = ol - yl
    # Calculates the Gradient between hidden layer and output layer 
    w2_gradiant = np.dot(np.transpose(delta_output), zj_bias) / n_train + (lambdaval / n_train) * w2
    # Calculates the Hidden layer error i.e., gradiant w.r.t hidden layer 
    delta_hidden = np.dot(delta_output, w2) * zj_bias * (1 - zj_bias)
    # Removing bias term from the hidden layer error 
    delta_hidden = delta_hidden[:, :-1]
    # Gradient for w1
    grad_w1 = np.dot(np.transpose(delta_hidden), training_data_bias) / n_train + (lambdaval / n_train) * w1
    # Flatten both the w1, w2 gradients into a single vector
    obj_grad = np.concatenate((grad_w1.flatten(), w2_gradiant.flatten()), 0)

    return (obj_val, obj_grad)

    


def nnPredict(w1, w2, data):
    """ nnPredict predicts the label of data given the parameters of the trained neural network.
    
    Input:
    w1: matrix of weights from input layer to hidden layer
    w2: matrix of weights from hidden layer to output layer
    data: matrix of data. Each row is a feature vector for one image.
    
    Output:
    labels: predicted labels for each data instance in the form of a 1D array
    """

    labels = np.array([])
    inp_data = data.shape[0]
    
    # Adding bias term to input data
    data_bias = np.concatenate([data, np.ones((inp_data, 1))], axis=1)

    # Forward Pass
    # Calculates the activation function 
    zj = sigmoid(np.dot(data_bias, np.transpose(w1)))   
    # Add bias term to hidden layer output
    zj_bias = np.concatenate((zj, np.ones((zj.shape[0], 1))), axis=1)
    # Output layer computations
    ol = sigmoid(np.dot(zj_bias, np.transpose(w2)))
    # Selecting the class with highest probability in the ouptut layer     
    labels = np.argmax(ol, axis=1)
    
    return labels



"""**************Neural Network Script Starts here********************************"""
start_time = time.time() 
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset
print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
end_time = time.time()
Training_time = end_time-start_time
print('\n Training Time:', Training_time)

with open('params.pickle', 'wb') as f:
    pickle.dump({
        "selected_features": selected_features,
        "w1": w1,
        "w2": w2,
        "lambdaval": lambdaval,
        "n_hidden": n_hidden
    }, f, protocol=3)

#with open('params.pickle', 'rb') as f:
#    params = pickle.load(f)
#print(params)


## ------------------------------HYPERPARAMETER TUNING---------------------------------------------
#start_time = time.time()
# Load and preprocess data
#train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
# Hyperparameter grid
# lambdaval_list = [0, 10, 20, 30, 40, 50, 60] 
# n_hidden_list = [4, 8, 12, 16, 20]

# training_times = np.zeros((len(lambdaval_list), len(n_hidden_list)))
# # Train and evaluate for each combination of lambdaval and n_hidden
# for i, lambdaval in enumerate(lambdaval_list):
#     for j, n_hidden in enumerate(n_hidden_list):
#         n_input = train_data.shape[1]
#         n_class = 10

#         # Initialize weights
#         initial_w1 = initializeWeights(n_input, n_hidden)
#         initial_w2 = initializeWeights(n_hidden, n_class)

#         # Unroll weights into a single vector
#         initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

#         # Set up args
#         args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#         # Train Neural Network using minimize function
#         opts = {'maxiter': 50}
#         nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

#         # Reshape weights
#         w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
#         w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#         # Predict labels for the training, validation, and test sets
#         predicted_label = nnPredict(w1, w2, train_data)
#         train_accuracy = 100 * np.mean((predicted_label == train_label).astype(float))

#         predicted_label = nnPredict(w1, w2, test_data)
#         test_accuracy = 100 * np.mean((predicted_label == test_label).astype(float))

#         predicted_label = nnPredict(w1, w2, validation_data)
#         validation_accuracy = 100 * np.mean((predicted_label == validation_label).astype(float))

#         end_time = time.time()
#         Training_time = end_time - start_time
#         print(f"n_hidden = {n_hidden}, lambdaval = {lambdaval}, Training set Accuracy: {train_accuracy:.2f}%, Test set Accuracy: {test_accuracy:.2f}%, Validation set Accuracy: {validation_accuracy:.2f}%, Training Time:{Training_time:.2f} seconds")

#         training_times[i, j] = Training_time


# # Plotting the graph with lambdaval on x-axis and n_hidden as legend
# plt.figure(figsize=(10, 6))

# for j, n_hidden in enumerate(n_hidden_list):
#     plt.plot(lambdaval_list, training_times[:, j], marker='o', linestyle='-', label=f'n_hidden={n_hidden}')

# plt.xlabel('Lambda Values (λ)')
# plt.ylabel('Training Time (seconds)')
# plt.title('Training Time vs. Lambda Values for Different Hidden Units')
# plt.legend(title='Number of Hidden Units')
# plt.grid(True)
# plt.show()