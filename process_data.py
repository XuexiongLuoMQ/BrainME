import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
# from torch_geometric.data import Data, Batch
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from feat import compute_x
import os
from functools import reduce
import random
import pickle as pkl
from torch_geometric.data import Data, Batch
# from torch.utils.data import TensorDataset


def load_data(data_path):
    data = sio.loadmat(data_path)
    data_name = os.path.basename(data_path).split('.')[0]

    thres = {'HIV':[0.6,0.01],'BP':[0.5,0.005],'PPMI':[0.01,0.01,0.01],'PD':[0.2,0.01]}
    if data_name == 'PPMI':
        m1 = []
        x = data['X']
        labels = data['label']
        n = x.shape[0]
        for i in range(n):
            m = x[i][0]
            m1.append(m[:,:,0].copy())
        m1 = np.stack(m1,2)
        graph = m1
    else:
            # fmri = data['fmri']
        fmri = data['fmri']
        labels = data['label']
        #labels = labels.T
            # graphs = [dti, fmri]
        graph = fmri
    print(graph.shape, labels.shape)
            
    labelenc = preprocessing.LabelEncoder()
    labels = labelenc.fit_transform(labels.flatten())
    print("label 0:", np.sum(labels==0), "label 1:", np.sum(labels==1))
        # print(labels.shape, labels)
    labels = torch.LongTensor(labels)
    #graph=graph.transpose(2,0,1)
    #cor_adj = Adjacency_KNN(graph)
    #adj = Binary_adjacency(cor_adj)
    adj = process_adj(graph, thres[data_name][1])
    adj_norm = normalize_adj(adj)
    x = process_feat(adj)
    dataset_list = []
    for i in range(adj.shape[0]):    
        graph_data = Data(x = x[i], adj=adj[i], adj_norm=adj_norm[i], labels = labels[i])
        dataset_list.append(graph_data)
    return dataset_list, labels
    
def Adjacency_KNN(fc_data: np.ndarray, k=5):
    if k == 0:
        return np.copy(fc_data)
    adjacency = np.zeros(fc_data.shape)
    for subject_idx, graph in enumerate(fc_data):
        topk_idx = np.argsort(graph)[:, -1:-k-1:-1]
        for row_idx, row in enumerate(graph):
            adjacency[subject_idx, row_idx, topk_idx[row_idx]] = row[topk_idx[row_idx]]
    for adj in adjacency:
      for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
          if adj[i][j] != 0:
            adj[j][i] = adj[i][j]
    return adjacency

def Binary_adjacency(cor_adjacency: np.array):
   bi_adjacency = np.zeros(cor_adjacency.shape)
   for subject_idx, graph in enumerate(cor_adjacency):
      for row in range(graph.shape[0]):
         for col in range(graph.shape[1]):
            bi_adjacency[subject_idx][row][col] = 1 if graph[row][col] != 0 else 0
   return bi_adjacency

def process_adj(A, threshold):
    A = np.transpose(A,(2,0,1))  # (N,N,n) -> (n,N,N)
    adj = (A>threshold) #* A
    mask = np.ones_like(adj)-np.expand_dims(np.eye(adj.shape[1]),axis=0)
    adj = adj*mask # remove diagonal
    adj = torch.from_numpy(adj).float()
    return adj

def normalize_adj(adjs):
    def _normalize(adj):
        adj = adj + torch.eye(adj.shape[0])
        rowsum = adj.sum(1)
        d_inv_sqrt = rowsum**(-0.5)
        d_inv_sqrt[torch.isinf(torch.Tensor(d_inv_sqrt))] = 0.
        d_mat_inv_sqrt = torch.diag(torch.Tensor(d_inv_sqrt))
        return d_mat_inv_sqrt @ torch.Tensor(adj) @ d_mat_inv_sqrt
    n = adjs.shape[0]
    adjs_norm = []
    for i in range(n):
        adjs_norm.append(_normalize(adjs[i]))
    adjs_norm = torch.stack(adjs_norm)
    return adjs_norm

def process_feat(adj):
    x = compute_x(torch.Tensor(adj), "adj")
    return x


class BatchDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def split_sup_que(data):
    n = len(data)
    inds = list(range(n))
    random.shuffle(inds)
    sup_num = round(n*2/3)
    sup_set = [data[inds[i]] for i in range(sup_num)]
    que_set = [data[inds[i]] for i in range(sup_num,n)]

    return pyDataset(sup_set), pyDataset(que_set)



def load_tasks(dataname=None, data=None, seed=42, n_task=1):
    """
    return 
    """
    assert dataname is not None or data is not None

    if dataname is not None:
        data_path = os.path.join('data', dataname, dataname+'.mat')
        data, _ = load_data(data_path)

    n = len(data)
    print("data size", len(data))
    inds = list(range(n))
    random.seed(seed)
    random.shuffle(inds)
    num = int(n/n_task)
    
    tasks = []
    for i in range(n_task):
        if i < n_task-1:
            task_idx = inds[i*num:(i+1)*num]
        else:
            task_idx = inds[i*num:]
        # import pdb; pdb.set_trace()

        tasks.append([data[i] for i in task_idx])
    
    task_sets = list(map(split_sup_que, tasks))
    return task_sets

def load_kfold(dataname, seed, kf=10):
    """
    return 
    """
    data_path = os.path.join('data', dataname+'.mat')
    data, labels = load_data(data_path)

    print("data size", len(data))
    random.seed(seed)
    tasks = []
    kf = KFold(n_splits=10, random_state=seed, shuffle=True)
    for train_index, test_index in kf.split(data, labels):
        tasks.append((data[train_index], labels[train_index], data[test_index], labels[test_index],))
    
    task_sets = list(map(skf_split, tasks))
    return task_sets

def skf_split(data_tuple):
    train_x, train_y, text_x, test_y = data_tuple
    # import pdb; pdb.set_trace()
    
    sup_set = BatchDataset(train_x, train_y)
    que_set = BatchDataset(text_x, test_y)

    print('support size', len(sup_set), 'query size', len(que_set))
    return sup_set, que_set

class pyDataset(Dataset):
  def __init__(self, data):
    # self.x = [item.x for item in data]
    # self.adj = [item.adj for item in data]
    # self.adj_norm = [item.adj_norm for item in data]
    # self.y = [item.labels for item in data]
    self.data = data
  
  def __getitem__(self, index):
    data = self.data[index]
    return {'x':data.x, 'adj':data.adj, 'adj_norm':data.adj_norm, 'y':data.labels}
    # return self.x[index], self.adj[index], self.adj_norm[index], self.y[index]
  
  def __len__(self):
    return len(self.data)

if __name__ == '__main__':
    tasks = load_tasks('HIV', 42)
    
    import pdb; pdb.set_trace()
