import numpy as np
from activations import relu_backward, sigmoid_backward

def linear_backward(dZ, cache, lambd):
	"""
	Implement the linear portion of backward propagation for a single layer (layer l)

	Arguments:
	dZ -- Gradient of the cost with respect to the linear output (of current layer l)
	cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	A_prev, W, b = cache
	m = A_prev.shape[1]
	
	dW = 1./m * np.dot(dZ, A_prev.T) + lambd * W/m 
	db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
	dA_prev = np.dot(W.T,dZ)
	
	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)
	
	return dA_prev, dW, db

def linear_activation_backward(dA, cache, lambd, activation, sparse_ae_parameters=()):
	"""
	Implement the backward propagation for the LINEAR->ACTIVATION layer.
	
	Arguments:
	dA -- post-activation gradient for current layer l 
	cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
	
	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	linear_cache, activation_cache = cache
	
	if activation == "relu":
		if sparse_ae_parameters:
			sparse_beta, rho, rho_hat = sparse_ae_parameters
			#print("dA1's shape:", dA.shape)
			#print("rho_hat's shape:", rho_hat.shape)
			dA = dA + sparse_beta * (- rho/rho_hat + (1-rho)/(1-rho_hat))
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
		
	elif activation == "sigmoid":
		if sparse_ae_parameters:
			sparse_beta, rho, rho_hat = sparse_ae_parameters
			#print("dA1's shape:", dA.shape)
			#print("rho_hat's shape:", rho_hat.shape)
			dA = dA + sparse_beta * (- rho/rho_hat + (1-rho)/(1-rho_hat))
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
	
	return dA_prev, dW, db

def L_model_backward(AL, Y, caches, loss_type = "cross", lambd = 0, sparse_ae_parameters = (), hidden_activation=None, 
						output_activation = "sigmoid"):
	"""
	Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
	
	Arguments:
	AL -- probability vector, output of the forward propagation (L_model_forward())
	Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
	caches -- list of caches containing:
				every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
				the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
	
	Returns:
	grads -- A dictionary with the gradients
			 grads["dA" + str(l)] = ... 
			 grads["dW" + str(l)] = ... 
			 grads["db" + str(l)] = ... 
	"""
	grads = {}
	L = len(caches) # the number of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
	
	# Initializing the backpropagation
	if loss_type == "square":
		dAL = - (Y - AL)
	# Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
		current_cache = caches[L-1]
		grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, lambd, output_activation)

	elif loss_type == "cross":
		dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
	# Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
		current_cache = caches[L-1]
		grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, lambd, output_activation)

	elif loss_type == "softmax":
		#dAL = - np.divide(Y, AL)
		dZL = AL - Y
		current_cache = caches[L-1]
		grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZL, current_cache[0], lambd)
	
	if hidden_activation:
		for l in reversed(range(L-1)):
			# lth layer: (RELU -> LINEAR) gradients.
			current_cache = caches[l]
			dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, lambd, hidden_activation, sparse_ae_parameters)
			grads["dA" + str(l)] = dA_prev_temp
			grads["dW" + str(l + 1)] = dW_temp# + lambd * current_cache[0][1]
			grads["db" + str(l + 1)] = db_temp

	return grads

	
	