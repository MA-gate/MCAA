import numpy as np
import get_data
import random
import tensorflow as tf
import time
import os


# ========================= data partition ==================================================================
def partition(graph_set, K_FOLD, k_fold):
	graph_set_ = {}
	for graph in graph_set:
		g_label = str(graph.label)
		if g_label not in graph_set_:
			graph_set_[g_label] = [graph]
		else:
			graph_set_[g_label].append(graph)

	graph_set_groups = {}
	proportion_a_g = 1/K_FOLD
	for gi in range(K_FOLD):
		graph_set_groups[gi] = []
		for g_label in graph_set_:
			sample_num_ = int(len(graph_set_[g_label])*proportion_a_g +0.5)
			index_s = sample_num_*gi
			index_e = sample_num_*(gi+1) if gi<K_FOLD-1 else len(graph_set_[g_label])
			graph_set_groups[gi] += graph_set_[g_label][index_s:index_e]
	# shuffle to avoid continue labels
	for gi in graph_set_groups:
		random.shuffle(graph_set_groups[gi])

	test_set = graph_set_groups[k_fold]
	train_set = []
	for gi in range(K_FOLD):
		if gi!=k_fold:
			train_set += graph_set_groups[gi]
	return train_set, test_set


# ==================  GCN_MCAA ===========================================================================

class GCN_MCAA(tf.keras.Model):
	def __init__(self, graph_set, train_set=None, test_set=None):
		super(GCN_MCAA, self).__init__()
		self.graph_set = graph_set
		self.train_set = train_set
		self.test_set = test_set
		self.input_size = self.graph_set[0].X.shape[1]
		self.output_size = self.graph_set[0].label.shape[1]
		self.edge_f_size = self.graph_set[0].A_attr.shape[2] if self.graph_set[0].A_attr is not None else 0
		self.cell_size = 15
		self.channel_size = 3
		self.layer_num = 2
		self.layer_num_MLP = 1
		self.activation_fun = tf.keras.activations.tanh
		self.outlayer_channel_size = 10
		self.outlayer_layer_num = 2
		self.outlayer_use_cnn = True
		self.outlayer_kernel_size = self.cell_size
		self.batch_size = 100
		self.epoch = 100	# default 100
		self.k_fold_num = K_FOLD
		self.test_part = 0
		self.write_log = True
		self.model_style = 'MAgate' # MA or MAgate

	def partition(self):
		if self.train_set is None:
			self.test_proportion = 1/self.k_fold_num
			test_sample_num = int(len(self.graph_set)*self.test_proportion)
			test_set_start_index = self.test_part*test_sample_num
			test_set_end_index = min((self.test_part+1)*test_sample_num, len(self.graph_set))
			self.test_set = self.graph_set[test_set_start_index:test_set_end_index]
			self.train_set = self.graph_set[0:test_set_start_index] + self.graph_set[test_set_end_index:]

		self.test_set_labels = []
		for graph in self.test_set:
			self.test_set_labels.append(graph.label)

	def init_weights(self):
		self.WEIGHTs_att = {}
		self.WEIGHTs_map = {}
		self.WEIGHTs_r = {}
		self.WEIGHTs_info = {}
		self.WEIGHTs_gate = {}
		if self.model_style=='MA':
			self.init_weights_MA()
		elif self.model_style=='MAgate':
			self.init_weights_MA()
			self.init_weights_gate()
		for li in range(self.layer_num):
			self.WEIGHTs_r[li] = {}
			for mi in range(self.layer_num_MLP):
				node_dim_in = self.input_size if li==0 else self.cell_size
				neibor_dim_in = self.input_size+self.edge_f_size if li==0 else self.cell_size+self.edge_f_size
				neibor_dim_out_MA = neibor_dim_in*self.channel_size
				neibor_dim_out_gate = self.cell_size
				if mi==0:
					if self.model_style=='MA':
						neibor_dim_out = neibor_dim_out_MA
					elif self.model_style=='MAgate':
						neibor_dim_out = neibor_dim_out_MA + neibor_dim_out_gate
					shape = (node_dim_in+neibor_dim_out, self.cell_size)
				else:
					shape = (self.cell_size, self.cell_size)
				self.WEIGHTs_r[li][mi] = self.add_weight(shape=shape,
														 initializer='truncated_normal',trainable=True)
		# output layer
		self.init_weights_outputLayer()

	def init_weights_MA(self):
		for li in range(self.layer_num):
			self.WEIGHTs_att[li] = {}
			shape = (self.input_size, self.channel_size*1) if li==0 else (self.cell_size, self.channel_size*1)
			self.WEIGHTs_att[li]['X0'] = self.add_weight(shape=shape,initializer='truncated_normal',trainable=True)
			self.WEIGHTs_att[li]['X1'] = self.add_weight(shape=shape,initializer='truncated_normal',trainable=True)
			if self.graph_set[0].A_attr is not None:
				shape = (self.edge_f_size, self.channel_size*1)
				self.WEIGHTs_att[li]['E'] = self.add_weight(shape=shape,initializer='truncated_normal',trainable=True)
	def init_weights_gate(self):
		for li in range(self.layer_num):
			self.WEIGHTs_info[li] = {}
			self.WEIGHTs_gate[li] = {}
			shape = (self.input_size, self.cell_size) if li==0 else (self.cell_size, self.cell_size)
			self.WEIGHTs_info[li]['X0'] = self.add_weight(shape=shape,initializer='truncated_normal',trainable=True)
			self.WEIGHTs_gate[li]['X0'] = self.add_weight(shape=shape,initializer='truncated_normal',trainable=True)
			self.WEIGHTs_info[li]['X1'] = self.add_weight(shape=shape,initializer='truncated_normal',trainable=True)
			self.WEIGHTs_gate[li]['X1'] = self.add_weight(shape=shape,initializer='truncated_normal',trainable=True)
			if self.graph_set[0].A_attr is not None:
				shape = (self.edge_f_size, self.cell_size)
				self.WEIGHTs_info[li]['E'] = self.add_weight(shape=shape,initializer='truncated_normal',trainable=True)
				self.WEIGHTs_gate[li]['E'] = self.add_weight(shape=shape,initializer='truncated_normal',trainable=True)


	def init_weights_outputLayer(self):
		# layer_output
		if self.outlayer_use_cnn:
			# attention_aggr_W for image
			shape=(self.input_size+self.cell_size*self.layer_num, self.outlayer_channel_size*1)
			self.WEIGHTs_out_att = self.add_weight(shape=shape, initializer='truncated_normal',trainable=True)
			# cnn 1dim kernels
			shape=(self.input_size+self.cell_size*self.layer_num, self.outlayer_kernel_size)
			self.WEIGHTs_out_cnn = self.add_weight(shape=shape, initializer='truncated_normal',trainable=True)
			# predict
			self.WEIGHTs_out_r = {}
			shape = (self.outlayer_kernel_size*self.outlayer_channel_size, self.output_size)
			self.WEIGHTs_out_r[0] = self.add_weight(shape=shape, initializer='truncated_normal',trainable=True)
		else:
			shape=(self.input_size+self.cell_size*self.layer_num, self.outlayer_channel_size*1)
			self.WEIGHTs_out_att = self.add_weight(shape=shape, initializer='truncated_normal',trainable=True)
			self.WEIGHTs_out_r = {}
			for i in range(self.outlayer_layer_num-1):
				if i==0: shape=((self.input_size+self.cell_size*self.layer_num)*self.outlayer_channel_size, self.cell_size)
				else: shape=(self.cell_size, self.cell_size)
				self.WEIGHTs_out_r[i] = self.add_weight(shape=shape, initializer='truncated_normal',trainable=True)
			shape=(self.cell_size, self.output_size)
			self.WEIGHTs_out_r[self.outlayer_layer_num-1] = self.add_weight(shape=shape, initializer='truncated_normal',trainable=True)


	def my_dense(self, x, w, activation_fun=None):
		if activation_fun:
			return activation_fun(tf.matmul(x, w) + 0.1)
		else:
			return tf.matmul(x, w) + 0.1

	def XEX_matmul_W(self, graph, layer_i, W):
		X, A, A_attr = graph.X if layer_i==0 else graph.X_new[layer_i-1], graph.A, graph.A_attr
		N_nodes, x_dim = X.shape
		Y0, Y1 = tf.matmul(X, W[layer_i]['X0']), tf.matmul(X, W[layer_i]['X1'])
		Y = tf.tile(tf.expand_dims(Y0,axis=1),[1,N_nodes,1]) + tf.tile(tf.expand_dims(Y1,axis=0),[N_nodes,1,1])
		if A_attr is not None:
			Y += tf.reshape(tf.matmul(tf.reshape(A_attr,[N_nodes*N_nodes,-1]), W[layer_i]['E']),[N_nodes,N_nodes,-1])
		# Y has a shape of [num_nodes, num_nodes, W_shape[1]]
		return Y

	def aggr_neibors_MA(self, graph, layer_i):
		X, A, A_attr = graph.X if layer_i==0 else graph.X_new[layer_i-1], graph.A, graph.A_attr
		N_nodes = X.shape[0]
		# 计算分数，将分数变为正数
		att = self.XEX_matmul_W(graph, layer_i, self.WEIGHTs_att)
		att = tf.keras.activations.elu(att) + 1.0001
		# 过滤不相邻节点
		att = tf.multiply(att, tf.tile(tf.expand_dims(A,axis=2), [1,1,self.channel_size]))
		# 计算 attention
		att = tf.nn.softmax(att, axis=1)
		# 过滤不相邻节点
		att = tf.multiply(att, tf.tile(tf.expand_dims(A,axis=2), [1,1,self.channel_size]))
		# aggregated neibors
		Y = tf.reshape(tf.matmul(tf.reshape(tf.transpose(att,[0,2,1]),[-1,N_nodes]), X), [N_nodes,-1])
		if A_attr is not None:
			Y_edge = tf.multiply(tf.tile(tf.expand_dims(att,axis=3),[1,1,1,A_attr.shape[2]]),
								 tf.tile(tf.expand_dims(A_attr,axis=2),[1,1,self.channel_size,1]))
			Y_edge = tf.reshape(tf.reduce_sum(Y_edge,axis=1),[N_nodes,-1])
			Y = tf.concat([Y_edge,Y], axis=1)
		#
		# Y = tf.multiply(Y, tf.expand_dims(tf.reduce_sum(A, axis=1),axis=1))*GRAD
		return Y

	def aggr_neibors_gate(self, graph, layer_i):
		info = tf.keras.activations.tanh(self.XEX_matmul_W(graph, layer_i, self.WEIGHTs_info))
		gate = tf.keras.activations.sigmoid(self.XEX_matmul_W(graph, layer_i, self.WEIGHTs_gate))
		Y = tf.multiply(info, gate)
		# aggregated neibors
		Y = tf.reduce_sum(Y, axis=1)
		return Y

	def update_X(self, graph, layer_i):
		X = graph.X if layer_i==0 else graph.X_new[layer_i-1]
		#
		if self.model_style=='MA':
			neibors = self.aggr_neibors_MA(graph, layer_i)
		elif self.model_style=='MAgate':
			neibors = tf.concat([self.aggr_neibors_MA(graph, layer_i),
								 self.aggr_neibors_gate(graph, layer_i)], axis=1)
		r = tf.concat([X, neibors], axis=1)
		for mi in range(self.layer_num_MLP):
			r = self.my_dense(r, self.WEIGHTs_r[layer_i][mi], activation_fun=self.activation_fun)
		graph.X_new[layer_i] = r

	def output_layer(self, graphset):
		if self.outlayer_use_cnn:
			for graph in graphset:
				X = tf.concat([graph.X, tf.reshape(tf.transpose(list(graph.X_new.values()),[1,0,2]),[graph.X.shape[0],-1])], axis=1)
				att = tf.matmul(X, self.WEIGHTs_out_att)
				att = tf.nn.softmax(att, axis=0)
				image = tf.matmul(tf.transpose(att,[1,0]), X) # shape: [outlayer_channelsize, x_dim_concated]
				# cnn kernels
				image = self.my_dense(image, self.WEIGHTs_out_cnn, activation_fun=self.activation_fun)
				# predict
				Y = self.my_dense(tf.reshape(image,[1,-1]), self.WEIGHTs_out_r[0], activation_fun=tf.nn.softmax)
				graph.pred_label = Y
		else:
			for graph in graphset:
				X = tf.concat([graph.X, tf.reshape(tf.transpose(list(graph.X_new.values()),[1,0,2]),[graph.X.shape[0],-1])], axis=1)
				att = tf.matmul(X, self.WEIGHTs_out_att)
				att = tf.nn.softmax(att, axis=0)
				Y = tf.reshape(tf.matmul(tf.transpose(att,[1,0]), X), [1,-1])
				for i in range(self.outlayer_layer_num-1):
					Y = self.my_dense(Y, self.WEIGHTs_out_r[i], activation_fun=self.activation_fun)
				graph.pred_label = self.my_dense(Y, self.WEIGHTs_out_r[self.outlayer_layer_num-1], activation_fun=tf.nn.softmax)

	def networks(self, graphset):
		for graph in graphset:
			graph.X_new = {}
		for layer_i in range(self.layer_num):
			for graph in graphset:
				self.update_X(graph, layer_i)
		# outputs
		self.output_layer(graphset)
		# Y
		self.real_Y = [graph.label for graph in graphset]
		self.pred_Y = [graph.pred_label for graph in graphset]
		self.real_Y = tf.reshape(self.real_Y, [-1, self.output_size])
		self.pred_Y = tf.reshape(self.pred_Y, [-1, self.output_size])


	def my_c_crossentropy_with_l2(self, real_y, pred_y):
		loss = tf.keras.losses.CategoricalCrossentropy()(real_y, pred_y)
		loss_regularizer = [tf.nn.l2_loss(p) for p in self.trainable_variables]
		loss_regularizer = tf.reduce_sum(loss_regularizer)/self._get_para_num()
		loss_ = loss + LAMBDA*loss_regularizer
		return loss_, loss, loss_regularizer

	def calcu_acc(self, real_y, pred_y):
		index_real = tf.argmax(real_y, axis=1)
		index_pred = tf.argmax(pred_y, axis=1)
		acc_li = tf.cast(tf.equal(index_real, index_pred), tf.int32)
		acc = tf.reduce_sum(acc_li) / tf.shape(acc_li)[0]
		return acc.numpy()

	def set2batches(self):
		self.train_batches = []
		self.test_batches = []
		batch_num = int(np.ceil(len(self.train_set)/self.batch_size))
		for batch_i in range(batch_num):
			s_index = batch_i * self.batch_size
			e_index = min((batch_i+1) * self.batch_size, len(self.train_set))
			self.train_batches.append(self.train_set[s_index:e_index])
		batch_num = int(np.ceil(len(self.test_set)/self.batch_size))
		for batch_i in range(batch_num):
			s_index = batch_i * self.batch_size
			e_index = min((batch_i+1) * self.batch_size, len(self.test_set))
			self.test_batches.append(self.test_set[s_index:e_index])

	def train(self, optimizer=None, loss_computer=None):
		start_time = time.process_time()
		string_settings = self._get_settings()
		print(string_settings)
		self._write_to_log(file_name=OUTPUT_FILE_epoch, string=string_settings+'\n')
		for epoch_i in range(self.epoch):
			results_epoch = []
			for batch in self.train_batches:
				with tf.GradientTape() as tape:
					self.networks(batch)
					loss, loss_pred, loss_regu = loss_computer(self.real_Y, self.pred_Y)
					acc = self.calcu_acc(self.real_Y, self.pred_Y)
					grads = tape.gradient(loss, self.trainable_variables)
					optimizer.apply_gradients(zip(grads, self.trainable_variables))
				train_rs = np.array([loss_pred.numpy().mean(), loss_regu.numpy().mean(), acc])
				# ************ test_set ***************************
				self.networks(self.test_set)
				loss_, loss_pred_, loss_regu_ = loss_computer(self.real_Y, self.pred_Y)
				acc_ = self.calcu_acc(self.real_Y, self.pred_Y)
				test_rs = np.array([loss_pred_.numpy().mean(), loss_regu_.numpy().mean(), acc_])
				# ************ output batch result ************
				rs = [train_rs[0],test_rs[0],train_rs[1],test_rs[1], train_rs[2], test_rs[2]]
				string = 'train/test loss, tr/test L2loss, tr/test acc:  %f  %f  %f  %f  %f  %f  use_time %.2f' \
						 % tuple(rs+[time.process_time()-start_time])
				print(string)
				results_epoch.append(rs)
			# ************ output epoch result ************
			string = '===== epoch %d finished =============='%(epoch_i)
			print(string)
			result_epoch = np.mean(results_epoch, axis=0)
			string = '%.4f   %.4f   %.6f   %.6f \t%.4f\t%.4f' %tuple(result_epoch)
			self._write_to_log(file_name=OUTPUT_FILE_epoch, string=string+'\n')
			# early stop
			train_loss, test_loss = result_epoch[0], result_epoch[1]
			if test_loss>train_loss*2:break

	def _write_to_log(self, file_name, string):
		if self.write_log:
			with open(file_name, 'a') as f:
				f.write(string)
	def _get_settings(self):
		string_settings = '\n==========================================================Model begin\n' + \
						'*** test_set_labels = '+str(np.sum(self.test_set_labels, axis=0)) + '\n' + \
						'*** input_size = '+str(self.input_size) + '\n' + \
						'*** input_size_edge = '+str(self.edge_f_size) + '\n' + \
						'*** cell_size = '+str(self.cell_size) + '\n' + \
						'*** channel_size = '+str(self.channel_size) + '\n' + \
						'*** layer_num = '+str(self.layer_num) + '\n' + \
						'*** layer_num_MLP = '+str(self.layer_num_MLP) + '\n' + \
						'*** outlayer_channel_size = '+str(self.outlayer_channel_size) + '\n' + \
						'*** outlayer_use_cnn = '+str(self.outlayer_use_cnn) + '\n' + \
						'*** outlayer_kernel_size = '+str(self.outlayer_kernel_size) + '\n' + \
						'*** outlayer_layer_num = '+str(self.outlayer_layer_num) + '\n' + \
						'*** activation_fun = '+str(self.activation_fun) + '\n' + \
						'*** test_part_k = '+str(self.test_part) + '\n' + \
						'*** batch_size = '+str(self.batch_size) + '\n' + \
						'*** epoch_num = '+str(self.epoch) + '\n' + \
						'*** paras_num = '+str(self._get_para_num()) + '\n' + \
						'*** model_style = '+str(self.model_style) + '\n' + \
						'*** LAMBDA = '+str(LAMBDA) + '\n' + \
						'==========================================='
		return string_settings
	def _get_para_num(self):
		num_params = 0
		for v in self.trainable_variables:
			shape = tf.shape(v).numpy()
			num_ = shape[0]
			for i in range(1,len(shape)):
				num_ *= shape[i]
			num_params += num_
		return num_params


	def run(self):
		self.init_weights()
		self.partition()
		self.set2batches()
		rs_test_acc = self.train(optimizer = tf.keras.optimizers.Adam(), loss_computer = self.my_c_crossentropy_with_l2)
		return rs_test_acc


# ***** Main *****
# =========================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)
# ====================== parameters =======================================
FILE_DIR = "XXXXX"
OUTPUT_FILE_epoch = FILE_DIR + "/results_epoch.txt"
K_FOLD = 10
LAMBDA = 6
test_fold = 0 # 0~9
# ============================ run =========================================
graph_set = get_data.get_data_PROTEINS()
train_set, test_set = partition(graph_set, K_FOLD, test_fold)
for try_ in range(3):
	model = GCN_MCAA(graph_set, train_set=train_set, test_set=test_set)
	model.test_part = test_fold
	model.run()
