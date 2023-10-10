import torch
import torch.nn as nn
import torch.nn.functional as F


# class MLP_subgraph(nn.Module):
#     def __init__(self,node_features_num, feature_size, num_nodes):
#         super(MLP_subgraph, self).__init__()
#         # self.device = device
#         self.node_features_num = node_features_num
#         # self.edge_features_num = edge_features_num
#         # self.mseloss = torch.nn.MSELoss()
#         self.feature_size = feature_size
#         self.linear = nn.Linear(self.node_features_num, self.feature_size)
#         self.linear1 = nn.Linear(2 * self.feature_size, 32)
#         # self.linear1 = nn.Linear(self.node_features_num * 2, 1).to(self.device)
#         self.linear2 = nn.Linear(32, 1)
#         self.num_nodes = num_nodes
        
    
#     def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
#         """
#         Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
#         """
#         if training:
#             bias = bias + 0.0001  # If bias is 0, we run into problems
#             eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)
#             gate_inputs = (torch.log(eps) - torch.log(1 - eps))
#             gate_inputs = gate_inputs.to(sampling_weights.device)
#             # print(f'\ntemperature{temperature.device}')
#             gate_inputs = (gate_inputs + sampling_weights) / temperature
#             graph =  torch.sigmoid(gate_inputs)
#         else:
#             graph = torch.sigmoid(sampling_weights)
#         return graph
    
#     # def concrete_sample(self, log_alpha, beta=1.0, training=True):
#     #     """ 
#     #     Sample from the instantiation of concrete distribution when training
#     #     \epsilon \sim  U(0,1), \hat{e}_{ij} = \sigma (\frac{\log \epsilon-\log (1-\epsilon)+\omega_{i j}}{\tau})
#     #     """
#     #     if training:
#     #         random_noise = torch.rand(log_alpha.shape)
#     #         random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
#     #         gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
#     #         gate_inputs = gate_inputs.sigmoid()
#     #     else:
#     #         gate_inputs = log_alpha.sigmoid()

#     #     return gate_inputs

#     def _edge_prob_mat(self, x):
#         # The number of nodes in the graph: node_num
#         # the number of features in the graph: feature_num
#         # graph = graph.to(self.device)
#         # edge_index = torch.nonzero(adj)
#         x = self.linear(x)
#         f1 = x.unsqueeze(1).repeat(1, self.num_nodes, 1).view(-1, self.feature_size)
#         f2 = x.unsqueeze(0).repeat(self.num_nodes, 1, 1).view(-1, self.feature_size)
#         f12self = torch.cat([f1, f2], dim=-1)
#         f12self = F.sigmoid(self.linear2(F.relu(self.linear1(f12self))))
#         mask_sigmoid = f12self.reshape(self.num_nodes, self.num_nodes)
#         sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
#         # edgemask = sym_mask[edge_index[0], edge_index[1]]
#         edgemask = self._sample_graph(sym_mask, temperature=0.5, bias=0.0, training=self.training)
#         return edgemask
#        positive_penalty = 0
#        adj_mask=[]
#        if torch.cuda.is_available():
#            EYE = torch.ones(0).cuda()
#        else:
#            EYE = torch.ones(0)
#            
#        for i in range(adj.shape[0]):

  
            #print(x[i],'99999999999')
#            abstract_features_1 = torch.tanh(self.linear1(x[i]))
#            assignment = torch.nn.functional.softmax(self.linear2(abstract_features_1), dim=1)
#            new_adj = torch.mm(torch.t(assignment),adj[i])
#            new_adj = torch.mm(new_adj,assignment)
#            normalize_new_adj = F.normalize(new_adj, p=1, dim=1)
#            norm_diag = torch.diag(adj[i])
            #print(norm_diag.shape,'11111')
#            pos_penalty = self.mseloss(normalize_new_adj,norm_diag)
#            positive_penalty += pos_penalty
#            adj_mask.append(assignment)
    

#     def forward(self, x):
#         # subgraph = graph.to(self.device)
#         edge_prob_matrix = self._edge_prob_mat(x)
#         # print(edge_prob_matrix.shape)
#         # subgraph.attr = edge_prob_matrix

#         # pos_penalty = edge_prob_matrix
#         sampled_adj = edge_prob_matrix
#         return sampled_adj

class MLP_subgraph(nn.Module):
    def __init__(self, inp_dim, hid_dim, num_nodes, tau=1.):
        super(MLP_subgraph, self).__init__()
        self.tau = tau
        self.linear1 = nn.Linear(inp_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, num_nodes)
        self.mseloss = torch.nn.MSELoss()
        # self.device = device


    def forward(self, x):

        
        #node_features = graph.node_features.to(self.device)
        #Adj = graph.adj.to(self.device) # graph adjacency matrix 
        batch_size, num_nodes, _ = x.shape

        node_feature_1 = F.relu(self.linear1(x)) 
        node_feature_2 =self.linear2(node_feature_1)

        node_mask = torch.sigmoid(node_feature_2)
        # import pdb; pdb.set_trace()
        node_group = node_mask.view(batch_size, num_nodes*num_nodes//2, 2) # reshape edge attention mask

        drop_mask = F.gumbel_softmax(node_group , tau=self.tau, hard=True) # Gumbel_softmax
        drop_mask = drop_mask.view(batch_size, num_nodes, num_nodes) #generate edge assignment
        
        #self.drop_mask_hard = self.drop_mask_hard * Adj 
        #edge = dense_to_sparse(self.drop_mask_hard)[0]
        
        #adj_mask=torch.stack(adj_mask)
       
        return drop_mask