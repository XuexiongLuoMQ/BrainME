import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = torch.matmul(adj,input).matmul(self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

class GCNEncoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_layer=2, graph=False):
        super(GCNEncoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        if n_layer == 2:
            self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout
        self.n_layer = n_layer
        self.graph = graph

    def forward(self, x, adj):
        if self.n_layer == 1:
            x = F.relu(self.gc1(x,adj))
        else:
            x = F.relu(self.gc1(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(self.gc2(x, adj))
        if self.graph:
            x = torch.mean(x,dim=1)
        return x

# if __name__ == '__main__':
#     gcn = GCNEncoder(90,16,0.1)
#     print(gcn.gc1.weight)