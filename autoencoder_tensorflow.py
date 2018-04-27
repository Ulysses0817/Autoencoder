from initializations import random_mini_batches
from tensorflow.python.framework import ops
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存

class autoencoder():
	
	#L = len(layer_dims)            # number of layers in the network
	#n_x, n_y = data_x.shape[0], data_y.shape[0]
		
	def __init__(self, isautoencoder = True):# layer_dims, learning_rate, training_epochs, batch_size, display_step, lambd, pool_f
		self.autoencoder = isautoencoder
		
	def create_placeholders(self, n_x, n_y):
		"""
		Creates the placeholders for the tensorflow session.

		Arguments:
		n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
		n_y -- scalar, number of classes (from 0 to 5, so -> 6)

		Returns:
		X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
		Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

		Tips:
		- You will use None because it let's us be flexible on the number of examples you will for the placeholders.
		  In fact, the number of examples during test/train is different.
		"""
		
		X = tf.placeholder(shape=[n_x, None],dtype='float')
		Y = tf.placeholder(shape=[n_y, None],dtype='float')
		
		return X, Y
	
	def initialize_parameters(self, layer_dims, lambd, initialize_type="Xavier", pythondict = None):
		"""
		Arguments:
		layer_dims -- python array (list) containing the dimensions of each layer in our network

		Returns:
		parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
						Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
						bl -- bias vector of shape (layer_dims[l], 1)
		"""

		#np.random.seed(1)
		parameters = {}
		L = len(layer_dims)            # number of layers in the network
		
		ininum = 2 if initialize_type == "He" else 1
		if pythondict == None:
			for l in range(1, L):
				name = 'W' + str(l)
				var = tf.Variable(tf.random_normal((layer_dims[l], layer_dims[l-1])) * np.sqrt(ininum/layer_dims[l-1]),
												  name=name) #, dtype=tf.float32
				tf.add_to_collection('ae_losses', tf.contrib.layers.l2_regularizer(lambd)(var))
#                 if l <= L//2 :
#                     tf.add_to_collection('soft_losses', tf.contrib.layers.l2_regularizer(lambd)(var))
				parameters[name] = var
				parameters['b' + str(l)] = tf.Variable(tf.zeros((layer_dims[l], 1)))# np.zeros((layer_dims[l], 1))
				
				assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
				assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
			
		else:
			for l in range(1, L):
				name = 'W' + str(l)
				var = tf.Variable(pythondict[name], name=name)#tf.convert_to_tensor(pythondict["W" + str(l)])
				tf.add_to_collection('ae_losses', tf.contrib.layers.l2_regularizer(lambd)(var))
				parameters[name] = var
				parameters["b" + str(l)] = tf.Variable(pythondict["b" + str(l)], name="b" + str(l))#tf.convert_to_tensor(pythondict["b" + str(l)])
				assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
				assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

		print("initialize_parameters——parameters.keys()————>", parameters.keys())
		return parameters
	
	def initialize_parameters_softmax(self, layer_dims, lambd, initialize_type="Xavier", pythondict = None):
		"""
		Arguments:
		layer_dims -- python array (list) containing the dimensions of each layer in our network

		Returns:
		parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
						Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
						bl -- bias vector of shape (layer_dims[l], 1)
		"""

		#np.random.seed(1)
		parameters = {}
		L = len(layer_dims)            # number of layers in the network
		
		ininum = 2 if initialize_type == "He" else 1
		if pythondict:
			for l in range(1, L-1):
				name = 'W' + str(l)
				var = tf.Variable(pythondict[name], name=name)#tf.convert_to_tensor(pythondict["W" + str(l)])
				tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))
				parameters[name] = var
				parameters["b" + str(l)] = tf.Variable(pythondict["b" + str(l)], name="b" + str(l))#tf.convert_to_tensor(pythondict["b" + str(l)])
				assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
				assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
		else:
			l = 0
			for l in range(1, L-1):
				name = 'W' + str(l)
				var = tf.Variable(tf.random_normal((layer_dims[l], layer_dims[l-1])) * np.sqrt(ininum/layer_dims[l-1]),
												  name=name) #, dtype=tf.float32
				tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))
				parameters[name] = var
				parameters['b' + str(l)] = tf.Variable(tf.zeros((layer_dims[l], 1)))# np.zeros((layer_dims[l], 1))
			
		l = l + 1 
		name = 'softmaxW'
		var = tf.Variable(tf.random_normal((layer_dims[l], layer_dims[l-1])) * np.sqrt(ininum/layer_dims[l-1]),
										  name=name) #, dtype=tf.float32
		tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))
		parameters[name] = var
		parameters['softmaxb'] = tf.Variable(tf.zeros((layer_dims[l], 1)))
		
		print("initialize_parameters_softmax——parameters.keys()————>", parameters.keys())
		return parameters
	
	
	def encoder(self, X, parameters, n_e, middle):
		"""
		Forward propagation part 1 -- compute the encoder output
		Arguments:
		X -- the input data of shape (the number of input units, the number of samples)
		parameters -- a dict contains "W1","b1"..."WL","bL"
		n_e -- the number of units in the last encoder layer
		pretain -- True or False
		"""
		A = X
		
		for l in range(1, middle+1):
			A_prev = A
			A = tf.nn.relu(tf.add(tf.matmul(parameters['W' + str(l)], A_prev), parameters['b' + str(l)]))
		print("the shape of encodered data—A————>", A.shape)
		
		assert(A.get_shape().as_list()[0] == n_e)
		
		return A
	
	def decoder(self, encoder_A, parameters, n_x, middle):
		"""
		Forward propagation part 1 -- compute the encoder output
		Arguments:
		encoder_A -- the output of the encoder part
		parameters -- a dict contains "W1","b1"..."WL","bL"
		n_x -- the number of input units
		pretain -- True or False
		"""
		A = encoder_A
		L = len(parameters) // 2
		
		for l in range(middle+1, L+1):
			A_prev = A
			A = tf.nn.relu(tf.add(tf.matmul(parameters['W' + str(l)], A_prev), parameters['b' + str(l)]))
		print("the shape of decodered data—A————>",A.shape)
		#print(A.get_shape().as_list()[0], n_x)
		assert(A.get_shape().as_list()[0] == n_x)
		
		return A
		
	def softmax(self, encoder_A, parameters):
		"""
		Compute the linear logits
		"""
		logits = tf.add(tf.matmul(parameters["softmaxW"], encoder_A), parameters["softmaxb"])
		
		return logits
	
	def compute_softmax_loss(self, A, Y, lambd=0.999):
		"""
		Computes the softmax loss
		Arguments:
		A -- output of forward propagation (output of the last unit), of shape (10, number of examples)
		Y -- "true" labels vector placeholder, same shape as A

		Returns:
		cost - Tensor of the cost function
		"""

		# to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
		logits = tf.transpose(A)
		labels = tf.transpose(Y)

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels))
		tf.add_to_collection('losses', cost)
		if lambd != 0.999:
			print("compute_softmax_loss——tf.get_collection('losses')————>", tf.get_collection('losses'))
			L2cost = tf.add_n(tf.get_collection('losses'))
			return L2cost
		
		else:
			return cost
		
	def kl_divergence(self, rho, rho_hat):
		# downL = tf.constant(1e-8)
		# rho_hat = tf.maximum(downL, rho_hat)
		return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1-rho_hat)
	
	def compute_cost(self, A, Y, lambd=0.999, rho=None, sparse_beta=None, A2=None):
		"""
		Computes the cost

		Arguments:
		A -- Output of forward propagation (output of the last unit), of shape (10, number of examples)
		Y -- "True" labels vector placeholder, same shape as A
		A2 -- Output of the hidden layer. In autoencoder, it indicates the output encoded data of 2nd layer
		lambd -- The coefficient of L2 Regularization term
		rho -- The Sparse Penalty level
		
		Returns:
		cost - Tensor of the cost function
		"""
		
		cost = tf.reduce_mean(tf.pow(A - Y, 2))
		L2_ = 0
		Sparse_ = 0
		
		#Calculate the L2 Regularization term
		if lambd != 0.999 :
			print("compute_cost——tf.get_collection('ae_losses')————>", tf.get_collection("ae_losses"))
			L2_ = tf.add_n(tf.get_collection('ae_losses'))

		#Calculate the Sparse penalty
		if rho :
			rho_hat = tf.reduce_mean(A2, axis=1)
			assert(rho_hat.get_shape().as_list()[0] == A2.get_shape().as_list()[0])
			kl_d = self.kl_divergence(rho, rho_hat)
			Sparse_ = sparse_beta * tf.reduce_sum(kl_d)
			
		cost = cost + L2_ + Sparse_			
		
		with tf.name_scope("Cost-components"):
			tf.summary.scalar('Mean-square-error', cost)
			tf.summary.scalar('L2-regularization-error', L2_)
			tf.summary.scalar('Sparsity-error', Sparse_)
			tf.summary.scalar('Minibatch-cost', cost)
			
		return cost
		
	def pooling(self, x, f, s, mode = "max"):
		"""
		Max pooling 
		Arguments:
		x -- The input data that will be pooled
		f -- The pooling size
		s -- stride
		mode -- Choose to use max() or mean()
		
		Return:
		pooled_x -- The pooled input data
		"""
		# 
		print(x.shape)
		#assert(x.shape[0] % f == 0)
		pool_size = int((x.shape[0]-f)/s + 1)
		print("pool_ed size:",pool_size)
		pooled_x = np.zeros((pool_size, x.shape[1]))
		p_l = 0
		for i in range(pool_size):

			p_r = p_l + f
			pooled_x[i, :] = np.max(x[p_l:p_r, :], axis=0) if mode == "max" else np.mean(x[p_l:p_r, :], axis=0)
			p_l = p_l + s
			
		print(pooled_x.shape)
		return pooled_x
	
	def py2tensor(self, pythondict, parameters):
		"""
		Convert python data type into tensor
		"""
		L = len(pythondict)//2
		for l in range(1, L+1):
			parameters["W" + str(l)] = tf.Variable(pythondict["W" + str(l)], name="W" + str(l))#tf.convert_to_tensor(pythondict["W" + str(l)])
			parameters["b" + str(l)] = tf.Variable(pythondict["b" + str(l)], name="b" + str(l))#tf.convert_to_tensor(pythondict["b" + str(l)])
			
		assert(len(parameters)//2 == int(L+1))
		return parameters
		
	def make_hparam_string(self, learning_rate, lambd, layer_dims, pool_f, stride, keep_prob, rho=0, sparse_beta=0):
		"""
		Create a string containing the hyperparameters we want
		"""
		rho = 0 if rho == None else rho
		sparse_beta = 0 if sparse_beta == None else sparse_beta
		lambd = 0 if lambd == 0.999 else lambd
		
		return "lr=%.0E,ld=%s,dims=%s,f-s=%s-%s,kp=%s,rho-sb=%s-%s-" % (learning_rate, lambd, layer_dims, pool_f, stride, keep_prob, rho, sparse_beta)
		
	def single_aemodel(self, train_x, test_x, layer_dims, num_ae = [1], learning_rate = 0.075, training_epochs = 1500, batch_size = 256, 
					   rho = None, display_step = 50, lambd = 0.3/256, pool_f = 0, initialize_type="Xavier", keep_prob = 1, 
					   print_cost = True, pythondict = None, decay_rate = 1, decay_steps = 4000, sparse_beta = None, stride = 0,
					   EB_labels_path = None):
		"""
		
		Arguments:
		decay_steps -- The number of times that the learning rate was decayed
		decay_rate -- The decay rate of the learning rate
		...
		
		Returns:
		(train_encoder, test_encoder) -- The tuple containing encoded train_x and encoded test_x
		ae_parameters -- Parameters containing the encoder and decoder parameters
		"""
		
		ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
		print("single_aemodel0——tf.get_collection('ae_losses')————>", tf.get_collection("ae_losses"))
		tf.set_random_seed(1)                             # to keep consistent results
		seed = 3                                          # to keep consistent results
		
		# Max-Pooling the input data
		if pool_f != 0 and (num_ae[0] == 1 or num_ae[0] == 0):
			train_x = self.pooling(train_x, pool_f, stride, "max")
			test_x = self.pooling(test_x, pool_f, stride, "max")
			
		(n_x, m) = train_x.shape                          # (n_x: input size, m : number of examples in the train set)
		
		# If pooled, adjust the layer's dimensions
		if pool_f != 0 and (num_ae[0] == 1 or num_ae[0] == 0):
			print("n_x————>", n_x)
			layer_dims[0], layer_dims[-1] = n_x, n_x
		
		#n_y = train_y.shape[0]                            # n_y : output size
		middle = len(layer_dims)//2    
		n_e = layer_dims[middle]     
		
		costs1 = []                                       # To keep track of the cost
		l_r = []				
		
		# Create Placeholders of shape (n_x, n_y)
		X = tf.placeholder(shape=[n_x, None], dtype='float')
		print("single_aemodel0——X————>", X)
		
		# Initialize parameters
		print("layers' dimensions:", layer_dims)
		parameters = self.initialize_parameters(layer_dims, lambd, initialize_type, pythondict)
		print("single_aemodel0——parameters————>", parameters)
		
		# Forward propagation: Build the forward propagation in the tensorflow graph
		encoder_op = self.encoder(X, parameters, n_e, middle)
		tf.summary.histogram('Hidden-layer-output', encoder_op)
		
		C_F = tf.transpose(self.encoder(X, parameters, n_e, middle))
		#print(C_F.get_shape().as_list())
		assert(C_F.get_shape().as_list()[1] == n_e)
		Compressed_Features = tf.Variable(tf.zeros([test_x.shape[1], n_e]), name="Compressed_Features")
		assignment = Compressed_Features.assign(C_F)
		
		AL = self.decoder(encoder_op, parameters, n_x, middle)
		print("single_aemodel0——AL————>", AL)
		# Cost function: Add cost function to tensorflow graph
		cost = self.compute_cost(AL, X, lambd, rho, sparse_beta, encoder_op)
		
		current_epoch = tf.Variable(tf.constant(0), trainable=False)
		#increment_current_epoch = tf.assign(current_epoch, current_epoch + 1)
		learning_rate = tf.train.exponential_decay(learning_rate, current_epoch, decay_steps, decay_rate, staircase=True)
		# Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
		
		#grads_and_vars = optimizer.compute_gradients(cost)
		train_op = optimizer.minimize(cost, global_step=current_epoch)#apply_gradients(grads_and_vars)#
		
		o = 0
		with tf.Session(config = config) as sess:
			saver = tf.train.Saver([Compressed_Features])
			
			sess.run(tf.global_variables_initializer())
			seed = 10
			print(parameters["b1"].eval()[:5])
			
			tf.summary.scalar("AdamOptimizer-learning-rate", optimizer._lr_t)
			
			# Merge all the summaries and write them out to /tmp
			merged = tf.summary.merge_all()
			daytime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())[5:16].replace(" ","-").replace(":","-" )
			params = self.make_hparam_string(learning_rate.eval(), lambd, layer_dims, pool_f, stride, keep_prob, rho, sparse_beta)
			EB_labels_path
			print(os.path.join(EB_labels_path, params))
			train_writer = tf.summary.FileWriter(os.path.join(EB_labels_path, params)+daytime+"-"+str(num_ae[0]), sess.graph)
			
			for epoch in range(training_epochs):
				
				# Loop over all batches
				epoch_cost = 0                       # Defines a cost related to an epoch
				num_minibatches = int(m / batch_size) # number of minibatches of size minibatch_size in the train set
				seed = seed + 1
				minibatches = random_mini_batches(train_x, train_x, batch_size, seed)
				
				for minibatch in minibatches:#for i in range(total_batch):
					
					(batch_xs, batch_ys) = minibatch #mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0

					if keep_prob != 1:
						# Mask some input data to improve the robustness
						D0 = np.random.rand(batch_xs.shape[0], batch_xs.shape[1])               # Step 1: initialize matrix D0 = np.random.rand(..., ...)
						D0 = D0 < keep_prob                                         # Step 2: convert entries of D0 to 0 or 1 (using keep_prob as the threshold)
						batch_xs = np.multiply(D0, batch_xs)                                    # Step 3: shut down some neurons of batch_xs(X)
						batch_xs = batch_xs/keep_prob                                             # Step 4: scale the value of neurons that haven't been shut down
					
					if num_ae[0] == 0 and o == 0:
						print("Original cost——>", sess.run(cost, feed_dict={X: batch_xs}))
						o = 1
					# Run optimization op (backprop) and cost op (to get loss value)
					_, minibatch_cost = sess.run([train_op, cost], feed_dict={X: batch_xs})
					epoch_cost += minibatch_cost / num_minibatches
					#print("Sparsity loss----->", np.squeeze(sloss), end="|")
				summary = sess.run(merged, feed_dict={X: batch_xs})
				train_writer.add_summary(summary, epoch)
				
				lr = sess.run(learning_rate, feed_dict={current_epoch:epoch})#
				# Display logs per epoch step
				costs1.append(epoch_cost)
				#l_r.append(lr)
				if epoch % display_step == 0 and print_cost == True:
					print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(epoch_cost),"learning rate tensor=","{:.9f}".format(sess.run(optimizer._lr_t)))
			
			sess.run(assignment, feed_dict={X: test_x})
			saver.save(sess, os.path.join('./logs/aetrain/', daytime+'.ckpt'), global_step=epoch+1)
			print("last cost——>", sess.run(cost, feed_dict={X: batch_xs}))
			print("Optimization Part{0} Finished!".format(num_ae[0]))
			train_writer.close()
			
			Config = projector.ProjectorConfig()
			# One can add multiple embeddings.
			#Config.model_checkpoint_path = os.path.join(EB_labels_path, "checkpoint")
			embedding = Config.embeddings.add()
			embedding.tensor_name = Compressed_Features.name
			# Link this tensor to its metadata file (e.g. labels).
			embedding.metadata_path = os.path.join(EB_labels_path, 'minstlabels.tsv')
			# Specify where you find the sprite (we will create this later)
			embedding.sprite.image_path = os.path.join(EB_labels_path, "sprite_1024.png") #'mnistdigits.png'
			embedding.sprite.single_image_dim.extend([28,28])
			# Saves a config file that TensorBoard will read during startup.
			projector.visualize_embeddings(tf.summary.FileWriter(EB_labels_path), Config)
			
			# plot the cost
			plt.figure(figsize=(10,7.5))
			plt.plot(np.squeeze(costs1))
			plt.ylabel('cost1')
			plt.xlabel('epoches (per tens)')
			plt.title("Learning rate =" + str(learning_rate))
			plt.show()
			
			train_encoder = sess.run(encoder_op, feed_dict={X:train_x})
			test_encoder = sess.run(encoder_op, feed_dict={X: test_x})
			test_pred = sess.run(AL, feed_dict={X: test_x})
			if num_ae[0] == 1 or num_ae[0] == 0:
				try:
					_ = np.reshape(test_x[:, 0], (28, 28))
					f, a = plt.subplots(2, 10, figsize=(20, 4))
					for i in range(10):
						a[0][i].imshow(np.reshape(test_x[:, i], (28, 28)))
						a[1][i].imshow(np.reshape(test_pred[:, i], (28, 28)))
					plt.show()
				except:
					pass
			
			# lets save the parameters in a variable
			print("Parameters have been trained!", parameters)
			parameters = sess.run(parameters)
			print(parameters["b1"][:5])
			ae_parameters, de_parameters = {}, {}
			if num_ae[0] == 0:
				
				return (train_encoder, test_encoder), parameters
			
			else:
				ae_parameters["W"+str(num_ae[0])] = parameters["W1"]
				ae_parameters["b"+str(num_ae[0])] = parameters["b1"]
				ae_parameters["W"+str(num_ae[1])] = parameters["W2"]
				ae_parameters["b"+str(num_ae[1])] = parameters["b2"]
				return (train_encoder, test_encoder), ae_parameters
#             print(tf.get_default_graph())
#             print(parameters)
#             print ("Parameters have been trained!")
#             L = len(layer_dims)
#             ae_parameters = {}
#             if self.autoencoder:
#                 print("Output AE parameters!")
#                 for l in range(1, (L-1)//2+1):
#                     ae_parameters['W' + str(l)] = parameters['W' + str(l)]
#                     ae_parameters['b' + str(l)] = parameters['b' + str(l)]
#                 print(ae_parameters)
#                 return ae_parameters, (train_x, train_Y, test_x, test_Y)
#             else:
#                 return parameters, (train_x, train_Y, test_x, test_Y)
	
	def stackedAE(self, train_x, test_x, layer_dims, learning_rate = [0.01, 0.000001], batch_size = 256, display_step = [50, 1], 
				  lambd = 0.3/256, training_epochs = [500, 20], pool_f = 0,keep_prob = 1, fine_tune = False, decay_rate = 0.96, 
				  decay_steps = 4000, rho = None, sparse_beta = None, stride = 0, EB_labels = None):
		"""
		Arguments:
		layer_dims -- python array (list) containing the dimensions of each layer in our AUTOENCODER network
		EB_labels -- Here it refers to test labels: (number of examples, 1)
		
		Reutrn:
		encoder_x -- Compressed features of the input data train_x
		"""
		assert(train_x.shape[0] == layer_dims[0])
		
		num_train = len(layer_dims)
		n_de = (num_train*2 - 2)
		
		encoder_x = (train_x, test_x)
		st_parameters, all_parameters = {},{}
		del_list = []
		
		LOG_DIR = "logs\\aetrain"
		ospath = os.getcwd()
		
		metadata = os.path.join(ospath, LOG_DIR)#.replace("\\", '/')
		print("labels' path:", metadata)
		# with open(metadata, 'w') as metadata_file:
			# metadata_file.write("Index\tLabel\n")
			# for row in EB_labels:
				# index = row[0]
				# label = row[1]
				# metadata_file.write("%d\t%d\n" % (index,label))
		
		for i in range(1, num_train):
			print("This is the {0}th training...".format(i))
			n_x = encoder_x[0].shape[0]
			n_e = layer_dims[i]
			train_epoch = training_epochs[i-1] if (i+1) <= len(training_epochs) else training_epochs[-2]
			l_r = learning_rate[i-1] if (i+1) <= len(learning_rate) else learning_rate[-2]
			display_s = display_step[0]
			encodered_train = encoder_x[0]
			encodered_test = encoder_x[1]
			encoder_x, encoder_p = self.single_aemodel(encodered_train, encodered_test, [n_x,n_e,n_x], [i, n_de], learning_rate=l_r, 
													   training_epochs = train_epoch, batch_size = batch_size, keep_prob = keep_prob,
													   display_step = display_s, lambd = lambd, pool_f = pool_f, decay_rate = decay_rate, 
													   decay_steps = decay_steps, rho = rho, sparse_beta = sparse_beta, stride = stride,
													   EB_labels_path = metadata)
#             st_parameters["W"+str(i)] = encoder_params["W"+str(i)]
#             st_parameters["b"+str(i)] = encoder_params["b"+str(i)]
			#print(type(all_parameters), type(encoder_p))
			print("*"*99)
			all_parameters.update(encoder_p)
			del_list.append("W"+str(n_de))
			del_list.append("b"+str(n_de))
			n_de = n_de - 1
		assert(n_de == i)
		print("All autoencoder parameters", all_parameters.keys())
		
		if fine_tune == True:
			display_s = display_step[1]
			train_epoch = training_epochs[-1]
			l_r = learning_rate[-1]
			ld = layer_dims.copy()
			ld.reverse()
			layer_dims_f = layer_dims + ld[1:]
			encoder_x, st_parameters = self.single_aemodel(train_x, test_x, layer_dims_f, [0, 0], learning_rate = l_r, batch_size = batch_size, 
														   training_epochs = train_epoch, display_step = display_s, lambd = lambd, 
														   pool_f = pool_f, pythondict=all_parameters, decay_rate = decay_rate, 
														   decay_steps = decay_steps, rho = rho, sparse_beta = sparse_beta, stride=stride, 
														   EB_labels_path = EB_labels)
		else:
			st_parameters = all_parameters
		for key in del_list:
			del st_parameters[key]

		return encoder_x, st_parameters
	
#         softmax_params = self.single_sfmodel(encodered_train, train_y, learning_rate = learning_rate, training_epochs = 500, batch_size = 256, 
#                                              display_step = 10, lambd = 0.3/256)
#         parameters["softmaxW"] = encoder_params["softmaxW"]
#         parameters["softmaxb"] = encoder_params["softmaxb"]
		
	def fineTune(self, train_x, train_y, test_x, test_y, layer_dims, learning_rate = 0.0001, training_epochs = 20, 
				pythondict = None, batch_size = 64, display_step = 1, lambd = 0.3/256, pool_f = 0, 
				initialize_type="Xavier", keep_prob = 1, print_cost = True, decay_rate = 0.96, 
				decay_steps = 4000, stride = 0):
		"""
		Take in the pretrained parameters and train the softmax layer
		
		Arguments:
		layer_dims -- python array (list) containing the dimensions of each layer in our network
		learning_rate -- the learning rate, scalar.
		batch_size -- the size of a mini batch
		
		Returns:
		parameters -- model's trained parameters
		
		"""
		print("Start to train softmax layer...")
		ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
		print(tf.get_collection("losses"))
		tf.set_random_seed(1)                             # to keep consistent results
		seed = 3                                          # to keep consistent results
		if pool_f != 0:
			# Max-Pooling the input data
			train_x = self.pooling(train_x, pool_f, stride, "max")
			test_x = self.pooling(test_x, pool_f, stride, "max")
			
		(n_x, m) = train_x.shape                          # (n_x: input size, m : number of examples in the train set)
		n_y = train_y.shape[0]                            # n_y : output size
		if pool_f != 0:
			layer_dims[0], layer_dims[-1] = n_x, n_y
			print(layer_dims)
		middle = len(layer_dims) - 2                      # middle : the index of the last encoder layer
		n_e = layer_dims[middle]                          # n_e : the number of the units in the last encoder layer
		
		costs2 = []                                       # To keep track of the cost
		accuracysTr, accuracysTe = [], []                 # To keep track of the accuracy
		
		# Create Placeholders of shape (n_x, n_y)
		X, Y = self.create_placeholders(n_x, n_y)
		print("X,Y———>", X, Y)
		
		# Initialize parameters
		parameters = self.initialize_parameters_softmax(layer_dims, lambd, initialize_type, pythondict)
		print("parameters———>", parameters)
		
		# Forward propagation: Build the forward propagation in the tensorflow graph
		encoder_op = self.encoder(X, parameters, n_e, middle) if middle > 0 else X
		tf.summary.histogram('Hidden-layer-output', encoder_op)
		# Compute the linear logits 
		AL = self.softmax(encoder_op, parameters)
		print("AL———>", AL)
		
		# Cost function: Add cost function to tensorflow graph
		loss = self.compute_softmax_loss(AL, Y, lambd)
		
		current_epoch = tf.Variable(tf.constant(0), trainable=False)
		learning_rate = tf.train.exponential_decay(learning_rate, current_epoch, decay_steps, decay_rate, staircase=True)
		# Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
		soft_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
		train_op = soft_optimizer.minimize(loss, global_step = current_epoch)
		
		# Calculate the correct predictions
		correct_prediction = tf.equal(tf.argmax(AL), tf.argmax(Y))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		cost_scalar = tf.summary.scalar('Classifying-epoch-cost', loss)
		accuracy_scalar = tf.summary.scalar('Accuracy', accuracy)
		
		with tf.Session(config = config) as sess:
			sess.run(tf.global_variables_initializer())
			
			if middle != 0:
				print("After global init——sess.run(parameters['b1'])[:10]", sess.run(parameters["b1"])[:10])
			seed = 10
			
			# Merge all the summaries and write them out to /tmp
			merged = tf.summary.merge_all()
			daytime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())[5:16].replace(" ","-").replace(":","-" )
			params = self.make_hparam_string(learning_rate.eval(), lambd, layer_dims, pool_f, stride, keep_prob)
			#params = "L_R-"+str(learning_rate) + "dims-" + str(layer_dims) + "ld-" + str(lambd) + "f-" + str(pool_f) + "s-" + str(stride) + "kp-"+ str(keep_prob)
			train_writer = tf.summary.FileWriter('./logs/ctrain/' + params + daytime, sess.graph)
			test_writer = tf.summary.FileWriter('./logs/ctest/' + params + daytime)
		
			for epoch in range(training_epochs):
				# Loop over all batches
				epoch_cost = 0.                       # Defines a cost related to an epoch
				num_minibatches = int(m / batch_size) # number of minibatches of size minibatch_size in the train set
				seed = seed + 1
				minibatches = random_mini_batches(train_x, train_y, batch_size, seed)
				
				summary, train_acc = sess.run([merged, accuracy],feed_dict={X: train_x, Y: train_y})
				acc_summary, test_acc = sess.run([accuracy_scalar, accuracy],feed_dict={X: test_x, Y: test_y})
				train_writer.add_summary(summary, epoch)
				test_writer.add_summary(acc_summary, epoch)
  
				for minibatch in minibatches: #for i in range(total_batch):
					(batch_xs, batch_ys) = minibatch
					
					if keep_prob != 1:
						# Mask some input data to improve the robustness
						D0 = np.random.rand(batch_xs.shape[0], batch_xs.shape[1])               # Step 1: initialize matrix D0 = np.random.rand(..., ...)
						D0 = D0 < keep_prob                                         # Step 2: convert entries of D0 to 0 or 1 (using keep_prob as the threshold)
						batch_xs = np.multiply(D0, batch_xs)                                    # Step 3: shut down some neurons of batch_xs(X)
						batch_xs = batch_xs/keep_prob                                             # Step 4: scale the value of neurons that haven't been shut down

					_, loss_ = sess.run([train_op, loss], feed_dict={X: batch_xs, Y: batch_ys})
					epoch_cost += loss_ / num_minibatches
				
				costs2.append(epoch_cost)
				accuracysTr.append(train_acc)
				accuracysTe.append(test_acc)
										 
				if epoch % display_step == 0 and print_cost == True:
					print("Epoch:", '%04d' % (epoch), "loss=", "{:.9f}".format(epoch_cost), 
						  "learning rate tensor=","{:.9f}".format(sess.run(soft_optimizer._lr_t)),
						  "Train Accuracy:", "{:.8f}".format(train_acc),
						  "Test Accuracy:", "{:.4f}".format(test_acc))
					
			print("Optimization Part2 Finished!")
			train_writer.close()
			test_writer.close()
			# plot the cost
			plt.figure(figsize=(10,7.5))
			plt.plot(np.squeeze(costs2))
			plt.plot(np.squeeze(accuracysTr))
			plt.plot(np.squeeze(accuracysTe))
			plt.ylabel('cost2')
			plt.xlabel('epoches (per tens)')
			plt.title("Learning rate =" + str(learning_rate))
			plt.show()
			
			print ("Train Accuracy:", accuracy.eval({X: train_x, Y: train_y}))
			print ("Test Accuracy:", accuracy.eval({X: test_x, Y: test_y}))
			
			return parameters

print("ok")

if __name__ == "__main__":
	ae = autoencoder()
	#encoder_x, aeparams = ae.stackedAE()