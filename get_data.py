import numpy as np
import pandas as pd


class Graph:
	def __init__(self):
		self.label = None
		self.A = None
		self.X = None
		self.X_neighbor_k = None
		self.X_neighbor_k_temp = None
		self.A_attr = None

def get_data_PROTEINS():
	print('=====> function get_data_PROTEINS().')
	file_A = 'XXX/dataset/PROTEINS/PROTEINS_A.txt'
	file_g_i = 'XXX/dataset/PROTEINS/PROTEINS_graph_indicator.txt'
	file_g_l = 'XXX/dataset/PROTEINS/PROTEINS_graph_labels.txt'
	file_n_l = 'XXX/dataset/PROTEINS/PROTEINS_node_labels.txt'
	file_n_a = 'XXX/dataset/PROTEINS/PROTEINS_node_attributes.txt'
	g_set = get_data(file_A=file_A, 
						 file_g_i=file_g_i, 
						 file_g_l=file_g_l, 
						 file_n_l=file_n_l,
						 file_n_a=file_n_a)
	return g_set

def get_data_MUTAG():
	print('=====> function get_data_MUTAG().')
	file_A = 'D:/dataset/MUTAG/MUTAG_A.txt'
	file_g_i = 'D:/dataset/MUTAG/MUTAG_graph_indicator.txt'
	file_g_l = 'D:/dataset/MUTAG/MUTAG_graph_labels.txt'
	file_n_l = 'D:/dataset/MUTAG/MUTAG_node_labels.txt'
	file_e_l = 'D:/dataset/MUTAG/MUTAG_edge_labels.txt'
	g_set = get_data(file_A=file_A,
						 file_g_i=file_g_i,
						 file_g_l=file_g_l,
						 file_n_l=file_n_l,
						 file_e_l=file_e_l)
	return g_set

def get_data_DD():
	print('=====> function get_data_DD().')
	file_A = 'D:/dataset/DD/DD_A.txt'
	file_g_i = 'D:/dataset/DD/DD_graph_indicator.txt'
	file_g_l = 'D:/dataset/DD/DD_graph_labels.txt'
	file_n_l = 'D:/dataset/DD/DD_node_labels.txt'
	g_set = get_data(file_A=file_A,
						 file_g_i=file_g_i,
						 file_g_l=file_g_l,
						 file_n_l=file_n_l)
	return g_set

def get_data_PTC():
	graph_set = []
	print('=====> function get_data_PTC().')
	file_A = 'XXX/dataset/PTC_MR/PTC_MR_A.txt'
	file_g_i = 'XXX/dataset/PTC_MR/PTC_MR_graph_indicator.txt'
	file_g_l = 'XXX/dataset/PTC_MR/PTC_MR_graph_labels.txt'
	file_n_l = 'XXX/dataset/PTC_MR/PTC_MR_node_labels.txt'
	file_e_l = 'XXX/dataset/PTC_MR/PTC_MR_edge_labels.txt'
	g_set = get_data(file_A=file_A,
						 file_g_i=file_g_i,
						 file_g_l=file_g_l,
						 file_n_l=file_n_l,
						 file_e_l=file_e_l)
	graph_set += g_set
	return graph_set

# def get_data_NC1():
# 	print('=====> function get_data_NC1().')
# 	file_A = 'XXX/dataset/NCI1/NCI1_A.txt'
# 	file_g_i = 'XXX/dataset/NCI1/NCI1_graph_indicator.txt'
# 	file_g_l = 'XXX/dataset/NCI1/NCI1_graph_labels.txt'
# 	file_n_l = 'XXX/dataset/NCI1/NCI1_node_labels.txt'
# 	g_set = get_data(file_A=file_A,
# 						 file_g_i=file_g_i,
# 						 file_g_l=file_g_l,
# 						 file_n_l=file_n_l)
# 	return g_set


def get_data(file_A=None,file_g_i=None,file_g_l=None,file_n_l=None,file_n_a=None,file_e_l=None,file_e_a=None):
	data_A = pd.read_csv(file_A, header=None)
	data_g_i = pd.read_csv(file_g_i, header=None)
	data_g_l = pd.read_csv(file_g_l, header=None)
	data_g_l = data_g_l.astype('object')
	data_g_l = to_vector(data_g_l, [0])
	data_n_l = pd.read_csv(file_n_l, header=None)
	data_n_l = data_n_l.astype('object')
	data_n_l = to_vector(data_n_l, [0])
	if file_n_a:
		data_n_a = pd.read_csv(file_n_a, header=None)
	if file_e_l:
		data_e_l = pd.read_csv(file_e_l, header=None)
		data_e_l = data_e_l.astype('object')
		data_e_l = to_vector(data_e_l, [0])
	if file_e_a:
		data_e_a = pd.read_csv(file_e_a, header=None)
	print('Num of graph_label:')
	print(data_g_l.sum())
	print('Num of nodes_in_graphs, sorteXXX')
	print(data_g_i.groupby([0]).size().sort_values(0).values, 'sum=',np.sum(data_g_i.groupby([0]).size().sort_values(0).values))
	print('shape of relation A:', data_A.shape)
	print('shape of node@graph:', data_g_i.shape)
	print('shape of graph_label:', data_g_l.shape)
	print('shape of node_label:', data_n_l.shape)
	if file_n_a:
		print('shape of node_attributes:', data_n_a.shape)
	if file_e_l:
		print('shape of edge_label:', data_e_l.shape)
	if file_e_a:
		print('shape of edge_attributes:', data_e_a.shape)

	# e_l & e_a concatenate to A
	if file_e_l:
		data_A = pd.concat([data_A,data_e_l], axis=1)
	if file_e_a:
		data_A = pd.concat([data_A,data_e_a], axis=1)
	print('shape of relation A, with edge label & edge attr:', data_A.shape)

	g_num = data_g_l.shape[0]
	graph_set = []
	for g_index in range(g_num):
		g = Graph()
		g_id = g_index + 1
		data_g_label = data_g_l.loc[g_index].values
		nodes_index = data_g_i[data_g_i[0]==g_id].index
		nodes_id = nodes_index + 1
		node_id_min = nodes_id[0]
		data_A_ = data_A[data_A[0].isin(nodes_id)].reset_index(drop=True)
		data_matrxA_ = np.zeros([len(nodes_id), len(nodes_id)])
		if file_e_l or file_e_a:
			data_matrxA_attr_ = np.zeros([len(nodes_id), len(nodes_id), data_A.shape[1]-2])

		for edge_i in range(data_A_.shape[0]):
			n_id1, n_id2 = data_A_.iloc[edge_i,0], data_A_.iloc[edge_i,1]
			i, j = n_id1-node_id_min, n_id2-node_id_min
			data_matrxA_[i, j] = 1
			data_matrxA_[j, i] = 1
			if file_e_l or file_e_a:
				e_attr = data_A_.iloc[edge_i,2:].values
				data_matrxA_attr_[i, j] = e_attr
				data_matrxA_attr_[j, i] = e_attr

		data_X = data_n_l.loc[nodes_index].values
		if file_n_a:
			data_n_a_ = data_n_a.loc[nodes_index].values
			data_X = np.concatenate([data_X, data_n_a_], axis=1)
		g.label = np.reshape(data_g_label, [1,-1]).astype(np.float32)
		g.A = data_matrxA_.astype(np.float32)
		g.X = data_X.astype(np.float32)
		if file_e_l or file_e_a:
			g.A_attr = data_matrxA_attr_.astype(np.float32)
		graph_set.append(g)
		if np.isnan(np.sum(g.X)) or np.isnan(np.sum(g.A)) or np.isnan(np.sum(g.label)):
			print('***** NaN *****')
		if file_e_l or file_e_a:
			if np.isnan(np.sum(g.A_attr)):
				print('***** NaN *****')

		if len(graph_set)%10==0:
			print('process of making Graphs %f' % (len(graph_set)/g_num))
	return graph_set


def to_vector(data, cate_col):
	data_cate = pd.get_dummies(data[cate_col])
	data = data.drop(cate_col, axis=1)
	data = pd.concat([data, data_cate], axis=1)
	return data


