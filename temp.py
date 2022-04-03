 # -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7)
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.core import Dense,Dropout,Activation
from tensorflow.python.keras.utils import np_utils

import itertools
import collections
from tqdm import tqdm
nb_classes = 10
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#从0-255 变到0-1
x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train,nb_classes)
y_test = np_utils.to_categorical(y_test,nb_classes)

def logistic(z): 
    return 1 / (1 + np.exp(-z))
 
def logistic_deriv(y):  # Derivative of logistic function
    return np.multiply(y, (1 - y))
 
def softmax(z): 
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
# Define the layers used in this model
class Layer(object):
    """Base class for the different layers.
    Defines base methods and documentation of methods."""
 
    def get_params_iter(self):
        """Return an iterator over the parameters (if any).
        The iterator has the same order as get_params_grad.
        The elements returned by the iterator are editable in-place."""
        return []
 
    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters.
        The list has the same order as the get_params_iter iterator.
        X is the input.
        output_grad is the gradient at the output of this layer.
        """
        return []
 
    def get_output(self, X):
        """Perform the forward step linear transformation.
        X is the input."""
        
 
    def get_input_grad(self, Y, output_grad=None, T=None):
        """Return the gradient at the inputs of this layer.
        Y is the pre-computed output of this layer (not needed in this case).
        output_grad is the gradient at the output of this layer 
         (gradient at input of next layer).
        Output layer uses targets T to compute the gradient based on the 
         output error instead of output_grad"""
        
class LinearLayer(Layer):
    """The linear layer performs a linear transformation to its input."""
 
    def __init__(self, n_in, n_out):
        """Initialize hidden layer parameters.
        n_in is the number of input variables.
        n_out is the number of output variables."""
        self.W = np.random.randn(n_in, n_out) * 0.1
        self.b = np.zeros(n_out)
 
    def get_params_iter(self):
        """Return an iterator over the parameters."""
        return itertools.chain(np.nditer(self.W, op_flags=['readwrite']),
                               np.nditer(self.b, op_flags=['readwrite']))
 
    def get_output(self, X):
        """Perform the forward step linear transformation."""
        return X.dot(self.W) + self.b
 
    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters."""
        JW = X.T.dot(output_grad)
        Jb = np.sum(output_grad, axis=0)
        return [g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))]
 
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return output_grad.dot(self.W.T)


class LogisticLayer(Layer):
    """The logistic layer applies the logistic function to its inputs."""
 
    def get_output(self, X):
        """Perform the forward step transformation."""
        return logistic(X)
 
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return np.multiply(logistic_deriv(Y), output_grad)
    
class SoftmaxOutputLayer(Layer):
    """The softmax output layer computes the classification propabilities at the output."""
 
    def get_output(self, X):
        """Perform the forward step transformation."""
        return softmax(X)
 
    def get_input_grad(self, Y, T):
        """Return the gradient at the inputs of this layer."""
        return (Y - T) / Y.shape[0]
 
    def get_cost(self, Y, T):
        """Return the cost at the output of this output layer."""
        return - np.multiply(T, np.log(Y)).sum() / Y.shape[0]
    
# Define a sample model to be trained on the data
hidden_neurons = 512  # Number of neurons in the first hidden-layer
#hidden_neurons_2 = 20  # Number of neurons in the second hidden-layer
# Create the model
layers = [] # Define a list of layers
# Add first hidden layer
layers.append(LinearLayer(x_train.shape[1], hidden_neurons))
layers.append(LogisticLayer())
# Add second hidden layer
#layers.append(LinearLayer(hidden_neurons_1, hidden_neurons_2))
#layers.append(LogisticLayer())
# Add output layer
layers.append(LinearLayer(hidden_neurons, y_train.shape[1]))
layers.append(SoftmaxOutputLayer())



# Define the forward propagation step as a method.
def forward_step(input_samples, layers):
    """
    Compute and return the forward activation of each layer in layers.
    Input:
        input_samples: A matrix of input samples (each row is an input vector)
        layers: A list of Layers
    Output:
        A list of activations where the activation at each index i+1 corresponds to
        the activation of layer i in layers. activations[0] contains the input samples.  
    """
    activations = [input_samples] # List of layer activations
    # Compute the forward activations for each layer starting from the first
    X = input_samples
    for layer in layers:
        Y = layer.get_output(X)  # Get the output of the current layer
        activations.append(Y)  # Store the output for future processing
        X = activations[-1]  # Set the current input as the activations of the previous layer
    return activations  # Return the activations of each layer


# Define the backward propagation step as a method
def backward_step(activations, targets, layers):
    """
    Perform the backpropagation step over all the layers and return the parameter gradients.
    Input:
        activations: A list of forward step activations where the activation at 
            each index i+1 corresponds to the activation of layer i in layers. 
            activations[0] contains the input samples. 
        targets: The output targets of the output layer.
        layers: A list of Layers corresponding that generated the outputs in activations.
    Output:
        A list of parameter gradients where the gradients at each index corresponds to
        the parameters gradients of the layer at the same index in layers. 
    """
    param_grads = collections.deque()  # List of parameter gradients for each layer
    output_grad = None  # The error gradient at the output of the current layer
    # Propagate the error backwards through all the layers.
    #  Use reversed to iterate backwards over the list of layers.
    for layer in reversed(layers):   
        Y = activations.pop()  # Get the activations of the last layer on the stack
        # Compute the error at the output layer.
        # The output layer error is calculated different then hidden layer error.
        if output_grad is None:
            input_grad = layer.get_input_grad(Y, targets)
        else:  # output_grad is not None (layer is not output layer)
            input_grad = layer.get_input_grad(Y, output_grad)
        # Get the input of this layer (activations of the previous layer)
        X = activations[-1]
        # Compute the layer parameter gradients used to update the parameters
        grads = layer.get_params_grad(X, output_grad)
        param_grads.appendleft(grads)
        # Compute gradient at output of previous layer (input of current layer):
        output_grad = input_grad
    return list(param_grads)  # Return the parameter gradients

# Create the minibatches
batch_size = 128  # Approximately 25 samples per batch
nb_of_batches = x_train.shape[0] / batch_size  # Number of batches
# Create batches (X,Y) from the training set
XT_batches = zip(
    np.array_split(x_train, nb_of_batches, axis=0),  # X samples
    np.array_split(y_train, nb_of_batches, axis=0))  # Y targets


# Define a method to update the parameters
def update_params(layers, param_grads, learning_rate):
    """
    Function to update the parameters of the given layers with the given gradients
    by gradient descent with the given learning rate.
    """
    for layer, layer_backprop_grads in zip(layers, param_grads):
        for param, grad in zip(layer.get_params_iter(), layer_backprop_grads):
            # The parameter returned by the iterator point to the memory space of
            #  the original layer and can thus be modified inplace.
            param -= learning_rate * grad  # Update each parameter
            
            

# Perform backpropagation
# initalize some lists to store the cost for future analysis        
minibatch_costs = []
training_costs = []
validation_costs = []
 
max_nb_of_iterations = 30  # Train for a maximum of 30 iterations
learning_rate = 0.1  # Gradient descent learning rate
 
# Train for the maximum number of iterations
for iteration in tqdm(range(max_nb_of_iterations)):
    for X, T in XT_batches:  # For each minibatch sub-iteration
        activations = forward_step(X, layers)  # Get the activations
        minibatch_cost = layers[-1].get_cost(activations[-1], T)  # Get cost
        minibatch_costs.append(minibatch_cost)
        param_grads = backward_step(activations, T, layers)  # Get the gradients
        update_params(layers, param_grads, learning_rate)  # Update the parameters
    # Get full training cost for future analysis (plots)
    activations = forward_step(x_train, layers)
    train_cost = layers[-1].get_cost(activations[-1], y_train)
    training_costs.append(train_cost)
    # Get full validation cost
    activations = forward_step(x_test, layers)
    validation_cost = layers[-1].get_cost(activations[-1], y_test)
    validation_costs.append(validation_cost)
    if len(validation_costs) > 3:
        # Stop training if the cost on the validation set doesn't decrease
        #  for 3 iterations
        if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
            break
nb_of_iterations = iteration + 1  # The number of iterations that have been executed
    