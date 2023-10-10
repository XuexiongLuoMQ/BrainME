import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data


class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None):
        """
        :param token_dim:
        :param token_num_per_group:
        :param group_num:   the total token number = token_num_per_group*group_num, in most cases, we let group_num=1.
                            In prompt_w_o_h mode for classification, we can let each class correspond to one group.
                            You can also assign each group as a prompt batch in some cases.

        :param prune_thre: if inner_prune is None, then all inner and cross prune will adopt this prune_thre
        :param isolate_tokens: if Trure, then inner tokens have no connection.
        :param inner_prune: if inner_prune is not None, then cross prune adopt prune_thre whereas inner prune adopt inner_prune
        """
        super(LightPrompt, self).__init__()

        self.inner_prune = inner_prune

        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])

        self.token_init(init_method="kaiming_uniform")

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self):
        return self.token_view()

    def token_view(self, ):
        """
        each token group is viewed as a prompt sub-graph.
        turn the all groups of tokens as a batch of prompt graphs.
        :return:
        """
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1

            # import pdb; pdb.set_trace()
            inner_adj = torch.where(token_sim < self.inner_prune, torch.zeros_like(token_sim), token_sim)
            # edge_index = inner_adj.nonzero().t().contiguous()

            # pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))
            pg_list.append((tokens, inner_adj, torch.tensor([i]).long()))

        # pg_batch = Batch.from_data_list(pg_list)

        return pg_list


class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.01):
        super(HeavyPrompt, self).__init__(token_dim, token_num, 1, inner_prune)  # only has one prompt graph.
        self.cross_prune = cross_prune

    def forward(self, x_batch):
        """
        TODO: although it recieves graph batch, currently we only implement one-by-one computing instead of batch computing
        TODO: we will implement batch computing once we figure out the memory sharing mechanism within PyG
        :param graph_batch:
        :return:
        """
        # device = torch.device("cuda")
        # device = torch.device("cpu")

        pg_x, pg_adj, _ = self.inner_structure_update()[0]  # batch of prompt graph (currently only 1 prompt graph in the batch)

        # inner_edge_index = pg.edge_index
        # token_num = pg.x.shape[0]

        # re_graph_list = []
        pg_x_batch = pg_x.unsqueeze(0).repeat(x_batch.shape[0], 1, 1)
        # import pdb; pdb.set_trace()
        cross_dot = torch.bmm(pg_x_batch, torch.transpose(x_batch, 1, 2))
        cross_sim = torch.sigmoid(cross_dot)
        cross_adj = torch.where(cross_sim < self.cross_prune, torch.zeros_like(cross_sim), cross_sim)

        # for i in range(len(x_batch)):
        #     # g_edge_index = g.edge_index + token_num
        #     # pg_x = pg.x.to(device)
        #     # g_x = g.x.to(device)
            
        #     cross_dot = torch.mm(pg_x, torch.transpose(x_batch[i], 0, 1))
        #     cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
        #     cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
            
        #     # cross_edge_index = cross_adj.nonzero().t().contiguous()
        #     # cross_edge_index[1] = cross_edge_index[1] + token_num
            
            
        #     # x = torch.cat([pg.x, g.x], dim=0)
        #     # y = g.y

        #     # edge_index = torch.cat([pg, g_edge_index, cross_edge_index], dim=1)
        #     # data = Data(x=x, edge_index=edge_index, y=y)
        #     # re_graph_list.append(data)
        #     re_graph_list.append(cross_adj)

        # graphp_batch = Batch.from_data_list(re_graph_list)
        # cross_batch = torch.stack(re_graph_list)  # (B,P,N)
        return pg_x, pg_adj, cross_adj
    
    