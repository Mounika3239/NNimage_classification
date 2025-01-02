'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
import time 
from math import sqrt
from scipy.optimize import minimize

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))# your code here
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
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
    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
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

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
start_time = time.time()
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
end_time = time.time()
Training_time = end_time-start_time

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
print('\n Training Time:', Training_time)


##------------------------------------HYPERPARAMETER TUNING---------------------------------------------

# start_time = time.time()
# train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
# # Hyperparameter grid
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
#       print(f"n_hidden = {n_hidden}, lambdaval = {lambdaval}, Training set Accuracy: {train_accuracy:.2f}%, Test set Accuracy: {test_accuracy:.2f}%, Validation set Accuracy: {validation_accuracy:.2f}%, Training Time:{Training_time:.2f} seconds")

#        training_times[i, j] = Training_time

#Plotting the graph 
#plt.figure(figsize=(10, 6))
#for i, lambdaval in enumerate(lambdaval_list):
#    plt.plot(n_hidden_list, training_times[i], marker='o', linestyle='-', label=f'λ={lambdaval}')

#plt.xlabel('Number of Hidden Units')
#plt.ylabel('Training Time in (seconds)')
#plt.title('Training Time vs. Number of Hidden Units')
#plt.legend(title='Lambdaval')
#plt.grid(True)
#plt.show()