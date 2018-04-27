import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import os
from forward import *
#os.makedirs("F:\\ProgramData\\NVIDIA Corporation\\CUDA Samples\\v9.0")
# def sigmoid(Z):
	# """
	# Implements the sigmoid activation in numpy
	
	# Arguments:
	# Z -- numpy array of any shape
	
	# Returns:
	# A -- output of sigmoid(z), same shape as Z
	# cache -- returns Z as well, useful during backpropagation
	# """
	
	# A = 1/(1+np.exp(-Z))
	# cache = Z
	
	# return A, cache

# def relu(Z):
	# """
	# Implement the RELU function.

	# Arguments:
	# Z -- Output of the linear layer, of any shape

	# Returns:
	# A -- Post-activation parameter, of the same shape as Z
	# cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
	# """
	
	# A = np.maximum(0,Z)
	
	# assert(A.shape == Z.shape)
	
	# cache = Z 
	# return A, cache


# def relu_backward(dA, cache):
	# """
	# Implement the backward propagation for a single RELU unit.

	# Arguments:
	# dA -- post-activation gradient, of any shape
	# cache -- 'Z' where we store for computing backward propagation efficiently

	# Returns:
	# dZ -- Gradient of the cost with respect to Z
	# """
	
	# Z = cache
	# dZ = np.array(dA, copy=True) # just converting dz to a correct object.
	
	# # When z <= 0, you should set dz to 0 as well. 
	# dZ[Z <= 0] = 0
	
	# assert (dZ.shape == Z.shape)
	
	# return dZ

# def sigmoid_backward(dA, cache):
	# """
	# Implement the backward propagation for a single SIGMOID unit.

	# Arguments:
	# dA -- post-activation gradient, of any shape
	# cache -- 'Z' where we store for computing backward propagation efficiently

	# Returns:
	# dZ -- Gradient of the cost with respect to Z
	# """
	
	# Z = cache
	
	# s = 1/(1+np.exp(-Z))
	# dZ = dA * s * (1-s)
	
	# assert (dZ.shape == Z.shape)
	
	# return dZ


def load_data():
	train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
	train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
	train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

	test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
	test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
	test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

	classes = np.array(test_dataset["list_classes"][:]) # the list of classes
	
	train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
	test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
	
	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# def initialize_parameters(n_x, n_h, n_y):
	# """
	# Argument:
	# n_x -- size of the input layer
	# n_h -- size of the hidden layer
	# n_y -- size of the output layer
	
	# Returns:
	# parameters -- python dictionary containing your parameters:
					# W1 -- weight matrix of shape (n_h, n_x)
					# b1 -- bias vector of shape (n_h, 1)
					# W2 -- weight matrix of shape (n_y, n_h)
					# b2 -- bias vector of shape (n_y, 1)
	# """
	
	# np.random.seed(1)
	
	# W1 = np.random.randn(n_h, n_x)*0.01
	# b1 = np.zeros((n_h, 1))
	# W2 = np.random.randn(n_y, n_h)*0.01
	# b2 = np.zeros((n_y, 1))
	
	# assert(W1.shape == (n_h, n_x))
	# assert(b1.shape == (n_h, 1))
	# assert(W2.shape == (n_y, n_h))
	# assert(b2.shape == (n_y, 1))
	
	# parameters = {"W1": W1,
				  # "b1": b1,
				  # "W2": W2,
				  # "b2": b2}
	
	# return parameters     

# # Xavier/He initialization
# def initialize_parameters_deep(layer_dims, intype="Xavier"):
	# """
	# Arguments:
	# layer_dims -- python array (list) containing the dimensions of each layer in our network
	
	# Returns:
	# parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
					# Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
					# bl -- bias vector of shape (layer_dims[l], 1)
	# """
	
	# np.random.seed(1)
	# parameters = {}
	# L = len(layer_dims)            # number of layers in the network
	
	# ininum = 2 if intype == "He" else 1
	# for l in range(1, L):
		# parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(ininum/layer_dims[l-1]) #*0.01
		# parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
		
		# assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
		# assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

		
	# return parameters

# # GRADED FUNCTION: random_mini_batches
# def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
	# """
	# Creates a list of random minibatches from (X, Y)
	
	# Arguments:
	# X -- input data, of shape (input size, number of examples)
	# Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
	# mini_batch_size -- size of the mini-batches, integer
	
	# Returns:
	# mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
	# """
	
	# np.random.seed(seed)            # To make your "random" minibatches the same as ours
	# m = X.shape[1]                  # number of training examples
	# mini_batches = []
		
	# # Step 1: Shuffle (X, Y)
	# permutation = list(np.random.permutation(m))
	# shuffled_X = X[:, permutation]
	# shuffled_Y = Y[:, permutation].reshape((1,m))

	# # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	# num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
	# for k in range(0, num_complete_minibatches):
		# ### START CODE HERE ### (approx. 2 lines)
		# mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
		# mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]
		# ### END CODE HERE ###
		# mini_batch = (mini_batch_X, mini_batch_Y)
		# mini_batches.append(mini_batch)
	
	# # Handling the end case (last mini-batch < mini_batch_size)
	# if m % mini_batch_size != 0:
		# ### START CODE HERE ### (approx. 2 lines)
		# mini_batch_X = shuffled_X[:, (k+1)*mini_batch_size:]
		# mini_batch_Y = shuffled_Y[:, (k+1)*mini_batch_size:]
		# ### END CODE HERE ###
		# mini_batch = (mini_batch_X, mini_batch_Y)
		# mini_batches.append(mini_batch)
	
	# return mini_batches
	
# # GRADED FUNCTION: initialize_velocity
# def initialize_velocity(parameters):
	# """
	# Initializes the velocity as a python dictionary with:
				# - keys: "dW1", "db1", ..., "dWL", "dbL" 
				# - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
	# Arguments:
	# parameters -- python dictionary containing your parameters.
					# parameters['W' + str(l)] = Wl
					# parameters['b' + str(l)] = bl
	
	# Returns:
	# v -- python dictionary containing the current velocity.
					# v['dW' + str(l)] = velocity of dWl
					# v['db' + str(l)] = velocity of dbl
	# """
	
	# L = len(parameters) // 2 # number of layers in the neural networks
	# v = {}
	
	# # Initialize velocity
	# for l in range(L):
		# v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
		# v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
		
	# return v
	
# # GRADED FUNCTION: initialize_adam
# def initialize_adam(parameters) :
	# """
	# Initializes v and s as two python dictionaries with:
				# - keys: "dW1", "db1", ..., "dWL", "dbL" 
				# - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
	
	# Arguments:
	# parameters -- python dictionary containing your parameters.
					# parameters["W" + str(l)] = Wl
					# parameters["b" + str(l)] = bl
	
	# Returns: 
	# v -- python dictionary that will contain the exponentially weighted average of the gradient.
					# v["dW" + str(l)] = ...
					# v["db" + str(l)] = ...
	# s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
					# s["dW" + str(l)] = ...
					# s["db" + str(l)] = ...

	# """
	
	# L = len(parameters) // 2 # number of layers in the neural networks
	# v = {}
	# s = {}
	
	# # Initialize v, s. Input: "parameters". Outputs: "v, s".
	# for l in range(L):
		# v["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
		# v["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
		# s["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
		# s["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
	
	# return v, s
	
# def linear_forward(A, W, b):
	# """
	# Implement the linear part of a layer's forward propagation.

	# Arguments:
	# A -- activations from previous layer (or input data): (size of previous layer, number of examples)
	# W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	# b -- bias vector, numpy array of shape (size of the current layer, 1)

	# Returns:
	# Z -- the input of the activation function, also called pre-activation parameter 
	# cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
	# """
	
	# Z = W.dot(A) + b
	
	# assert(Z.shape == (W.shape[0], A.shape[1]))
	# cache = (A, W, b)
	
	# return Z, cache

# def linear_activation_forward(A_prev, W, b, activation):
	# """
	# Implement the forward propagation for the LINEAR->ACTIVATION layer

	# Arguments:
	# A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
	# W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	# b -- bias vector, numpy array of shape (size of the current layer, 1)
	# activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

	# Returns:
	# A -- the output of the activation function, also called the post-activation value 
	# cache -- a python dictionary containing "linear_cache" and "activation_cache";
			 # stored for computing the backward pass efficiently
	# """
	
	# if activation == "sigmoid":
		# # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		# Z, linear_cache = linear_forward(A_prev, W, b)
		# A, activation_cache = sigmoid(Z)
	
	# elif activation == "relu":
		# # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		# Z, linear_cache = linear_forward(A_prev, W, b)
		# A, activation_cache = relu(Z)
	
	# assert (A.shape == (W.shape[0], A_prev.shape[1]))
	# cache = (linear_cache, activation_cache)

	# return A, cache

# def L_model_forward(X, parameters):
	# """
	# Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
	
	# Arguments:
	# X -- data, numpy array of shape (input size, number of examples)
	# parameters -- output of initialize_parameters_deep()
	
	# Returns:
	# AL -- last post-activation value
	# caches -- list of caches containing:
				# every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
				# the cache of linear_sigmoid_forward() (there is one, indexed L-1)
	# """

	# caches = []
	# A = X
	# L = len(parameters) // 2                  # number of layers in the neural network
	
	# # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
	# for l in range(1, L):
		# A_prev = A 
		# A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
		# caches.append(cache)
	
	# # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
	# AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
	# caches.append(cache)
	
	# assert(AL.shape == (1,X.shape[1]))
	
	# return AL, caches

# def compute_cost(AL, Y):
	# """
	# Implement the cost function defined by equation (7).

	# Arguments:
	# AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
	# Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

	# Returns:
	# cost -- cross-entropy cost
	# """
	
	# m = Y.shape[1]
	
	# # Compute loss from aL and y.
	# cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
	
	# cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	# assert(cost.shape == ())
	
	# return cost

# # GRADED FUNCTION: compute_cost_with_regularization
# def compute_cost(AL, Y, parameters=0, lambd=0):
	# """
	# Implement the cost function with L2 regularization. See formula (2) above.
	
	# Arguments:
	# A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
	# Y -- "true" labels vector, of shape (output size, number of examples)
	# parameters -- python dictionary containing parameters of the model
	
	# Returns:
	# cost - value of the regularized loss function (formula (2))
	# """
	# m = Y.shape[1]
	
	# # Compute loss from aL and y.
	# cross_entropy_cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
	
	# cross_entropy_cost = np.squeeze(cross_entropy_cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	# assert(cross_entropy_cost.shape == ())
	
	# L2_regularization_sum = 0
	
	# if lambd != 0:
		# L = len(parameters)//2
		# for l in range(L):
			# L2_regularization_sum = L2_regularization_cost + np.sum(np.square(parameters["W"+str(l+1)])) # L2_regularization_sum equals to W^1 + ... +W^L
	# L2_regularization_cost = L2_regularization_sum * lambd/(2*m)
	
	# cost = cross_entropy_cost + L2_regularization_cost
	
	# return cost

# def linear_backward(dZ, cache):
	# """
	# Implement the linear portion of backward propagation for a single layer (layer l)

	# Arguments:
	# dZ -- Gradient of the cost with respect to the linear output (of current layer l)
	# cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

	# Returns:
	# dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	# dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	# db -- Gradient of the cost with respect to b (current layer l), same shape as b
	# """
	# A_prev, W, b = cache
	# m = A_prev.shape[1]

	# dW = 1./m * np.dot(dZ,A_prev.T)
	# db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
	# dA_prev = np.dot(W.T,dZ)
	
	# assert (dA_prev.shape == A_prev.shape)
	# assert (dW.shape == W.shape)
	# assert (db.shape == b.shape)
	
	# return dA_prev, dW, db

# def linear_activation_backward(dA, cache, activation):
	# """
	# Implement the backward propagation for the LINEAR->ACTIVATION layer.
	
	# Arguments:
	# dA -- post-activation gradient for current layer l 
	# cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
	# activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
	
	# Returns:
	# dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	# dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	# db -- Gradient of the cost with respect to b (current layer l), same shape as b
	# """
	# linear_cache, activation_cache = cache
	
	# if activation == "relu":
		# dZ = relu_backward(dA, activation_cache)
		# dA_prev, dW, db = linear_backward(dZ, linear_cache)
		
	# elif activation == "sigmoid":
		# dZ = sigmoid_backward(dA, activation_cache)
		# dA_prev, dW, db = linear_backward(dZ, linear_cache)
	
	# return dA_prev, dW, db

# def L_model_backward(AL, Y, caches):
	# """
	# Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
	
	# Arguments:
	# AL -- probability vector, output of the forward propagation (L_model_forward())
	# Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
	# caches -- list of caches containing:
				# every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
				# the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
	
	# Returns:
	# grads -- A dictionary with the gradients
			 # grads["dA" + str(l)] = ... 
			 # grads["dW" + str(l)] = ...
			 # grads["db" + str(l)] = ... 
	# """
	# grads = {}
	# L = len(caches) # the number of layers
	# m = AL.shape[1]
	# Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
	
	# # Initializing the backpropagation
	# dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
	
	# # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
	# current_cache = caches[L-1]
	# grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
	
	# for l in reversed(range(L-1)):
		# # lth layer: (RELU -> LINEAR) gradients.
		# current_cache = caches[l]
		# dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
		# grads["dA" + str(l)] = dA_prev_temp
		# grads["dW" + str(l + 1)] = dW_temp
		# grads["db" + str(l + 1)] = db_temp

	# return grads

# def update_parameters(parameters, grads, learning_rate):
	# """
	# Update parameters using gradient descent
	
	# Arguments:
	# parameters -- python dictionary containing your parameters 
	# grads -- python dictionary containing your gradients, output of L_model_backward
	
	# Returns:
	# parameters -- python dictionary containing your updated parameters 
				  # parameters["W" + str(l)] = ... 
				  # parameters["b" + str(l)] = ...
	# """
	
	# L = len(parameters) // 2 # number of layers in the neural network

	# # Update rule for each parameter. Use a for loop.
	# for l in range(L):
		# parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
		# parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
		
	# return parameters

# # GRADED FUNCTION: update_parameters_with_momentum
# def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
	# """
	# Update parameters using Momentum
	
	# Arguments:
	# parameters -- python dictionary containing your parameters:
					# parameters['W' + str(l)] = Wl
					# parameters['b' + str(l)] = bl
	# grads -- python dictionary containing your gradients for each parameters:
					# grads['dW' + str(l)] = dWl
					# grads['db' + str(l)] = dbl
	# v -- python dictionary containing the current velocity:
					# v['dW' + str(l)] = ...
					# v['db' + str(l)] = ...
	# beta -- the momentum hyperparameter, scalar
	# learning_rate -- the learning rate, scalar
	
	# Returns:
	# parameters -- python dictionary containing your updated parameters 
	# v -- python dictionary containing your updated velocities
	# """

	# L = len(parameters) // 2 # number of layers in the neural networks
	
	# # Momentum update for each parameter
	# for l in range(L):
		
		# #(approx. 4 lines)
		# # compute velocities
		# v["dW" + str(l+1)] = beta*v["dW" + str(l+1)] + (1 - beta)*grads["dW" + str(l+1)]
		# v["db" + str(l+1)] = beta*v["db" + str(l+1)] + (1 - beta)*grads["db" + str(l+1)]
		# # update parameters
		# parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v["dW" + str(l+1)]
		# parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v["db" + str(l+1)]
		
		
	# return parameters, v

# # GRADED FUNCTION: update_parameters_with_adam

# def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
								# beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
	# """
	# Update parameters using Adam
	
	# Arguments:
	# parameters -- python dictionary containing your parameters:
					# parameters['W' + str(l)] = Wl
					# parameters['b' + str(l)] = bl
	# grads -- python dictionary containing your gradients for each parameters:
					# grads['dW' + str(l)] = dWl
					# grads['db' + str(l)] = dbl
	# v -- Adam variable, moving average of the first gradient, python dictionary
	# s -- Adam variable, moving average of the squared gradient, python dictionary
	# learning_rate -- the learning rate, scalar.
	# beta1 -- Exponential decay hyperparameter for the first moment estimates 
	# beta2 -- Exponential decay hyperparameter for the second moment estimates 
	# epsilon -- hyperparameter preventing division by zero in Adam updates

	# Returns:
	# parameters -- python dictionary containing your updated parameters 
	# v -- Adam variable, moving average of the first gradient, python dictionary
	# s -- Adam variable, moving average of the squared gradient, python dictionary
	# """
	
	# L = len(parameters) // 2                 # number of layers in the neural networks
	# v_corrected = {}                         # Initializing first moment estimate, python dictionary
	# s_corrected = {}                         # Initializing second moment estimate, python dictionary
	
	# # Perform Adam update on all parameters
	# for l in range(L):
		# # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
		# v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)] + (1-beta1)*grads["dW" + str(l+1)]
		# v["db" + str(l+1)] = beta1*v["db" + str(l+1)] + (1-beta1)*grads["db" + str(l+1)]

		# # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
		# v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-beta1**t)
		# v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-beta1**t)

		# # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
		# s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)] + (1-beta2)*grads["dW" + str(l+1)]**2
		# s["db" + str(l+1)] = beta2*s["db" + str(l+1)] + (1-beta2)*grads["db" + str(l+1)]**2

		# # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
		# s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-beta2**t)
		# s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1-beta2**t)      

		# # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
		# parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v_corrected["dW" + str(l+1)]/(np.sqrt(s_corrected["dW" + str(l+1)])+epsilon)
		# parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v_corrected["db" + str(l+1)]/(np.sqrt(s_corrected["db" + str(l+1)])+epsilon)
		
	# return parameters, v, s
	
def predict(X, y, parameters, hidden_activation=None, output_activation="sigmoid"):
	"""
	This function is used to predict the results of a  L-layer neural network.
	
	Arguments:
	X -- data set of examples you would like to label
	parameters -- parameters of the trained model
	
	Returns:
	p -- predictions for the given dataset X
	"""
	
	m = X.shape[1]
	n = len(parameters) // 2 # number of layers in the neural network
	p = np.zeros((1,m))
	
	# Forward propagation
	probas, caches = L_model_forward(X, parameters, 1, hidden_activation, output_activation)
	
	# If the activation of the output layer is "sigmoid", compute the accuracy like this.
	if output_activation == "sigmoid":
		# convert probas to 0/1 predictions
		for i in range(0, probas.shape[1]):
			if probas[0,i] > 0.5:
				p[0,i] = 1
			else:
				p[0,i] = 0
		#print results
		#print ("predictions: " + str(p))
		#print ("true labels: " + str(y))
		print("Accuracy: "  + str(np.sum((p == y)/m)))
		return (probas>0.5)
	elif output_activation == "softmax":
		# convert probas to 0/
		prob_max_index = np.argmax(probas, axis=0)
		pred_true = np.argmax(y, axis=0)
		accuracy = np.sum((prob_max_index==pred_true))/m
		print("Accuracy: ", accuracy)
		return prob_max_index
		
def print_mislabeled_images(classes, X, y, p):
	"""
	Plots images where predictions and truth were different.
	X -- dataset
	y -- true labels
	p -- predictions
	"""
	a = p + y
	mislabeled_indices = np.asarray(np.where(a == 1))
	plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
	num_images = len(mislabeled_indices[0])
	for i in range(num_images):
		index = mislabeled_indices[1][i]
		
		plt.subplot(2, num_images, i + 1)
		plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
		plt.axis('off')
		plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
