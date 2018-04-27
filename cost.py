import numpy as np

def compute_cost_raw(AL, Y):
	"""
	Implement the cost function defined by equation (7).

	Arguments:
	AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
	Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

	Returns:
	cost -- cross-entropy cost
	"""
	
	m = Y.shape[1]
	
	# Compute loss from aL and y.
	cost = (1./(2.*m))*np.sum((Y - AL)**2)#(1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
	
	cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	assert(cost.shape == ())
	
	return cost

# GRADED FUNCTION: compute_cost_with_regularization
def compute_cost(AL, Y, loss_type='cross', parameters={}, lambd=0, sparse_ae_parameters=()):
	"""
	Implement the cost function with L2 regularization. See formula (2) above.
	
	Arguments:
	AL -- post-activation, output of forward propagation, of shape (output size, number of examples)
	Y -- "true" labels vector, of shape (output size, number of examples)
	parameters -- python dictionary containing parameters of the model
	
	Returns:
	cost - value of the regularized loss function (formula (2))
	"""
	m = Y.shape[1]
	Jsparse = 0
	cross_entropy_cost = 0
	mean_squared_cost = 0
	L2_regularization_cost = 0
	softmax_cost = 0
	
	if loss_type == 'cross':
		# Compute cross-entropy from aL and y.
		cross_entropy_cost = (1./m) * np.sum((- np.dot(Y, np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T)))
		#print(cross_entropy_cost)
		cross_entropy_cost = np.squeeze(cross_entropy_cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
		#print(cross_entropy_cost)
		assert(cross_entropy_cost.shape == ())
		
	elif loss_type == 'square':
		# Compute squared cost from aL and y.
		mean_squared_cost = (1./(2.*m))*np.sum((Y - AL)**2)
		mean_squared_cost = np.squeeze(mean_squared_cost)
		assert(mean_squared_cost.shape == ())
		
	elif loss_type == 'softmax':
		softmax_cost = - (1./m) * np.sum(Y * np.log(AL))
		assert(softmax_cost.shape == ())
	
	if lambd != 0:
		# Compute L2 regularization's cost from aL and y.
		L = len(parameters)//2
		L2_regularization_sum = 0
		for l in range(L):
			L2_regularization_sum = L2_regularization_sum + np.sum(np.square(parameters["W"+str(l+1)])) # L2_regularization_sum equals to W^1 + ... +W^L
		L2_regularization_cost = L2_regularization_sum * lambd/(2*m)
	
	if sparse_ae_parameters:
		# compute the sparse cost for a1
		sparse_beta, rho, rho_hat = sparse_ae_parameters
		#print(rho/rho_hat)
		Jsparse = np.sum(rho * np.log(rho/rho_hat) + (1-rho) * np.log((1-rho)/(1-rho_hat)))
		Jsparse = sparse_beta * Jsparse

	#print(cross_entropy_cost, mean_squared_cost, L2_regularization_cost, Jsparse)
	cost = cross_entropy_cost + softmax_cost + mean_squared_cost + L2_regularization_cost + Jsparse
	
	return cost
