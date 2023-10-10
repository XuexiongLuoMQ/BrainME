from torch import nn, optim
import torch
from copy import deepcopy
from utils import seed_everything, node_score_batch, gib_loss,entropy_node_score
from random import shuffle
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from process_data import load_tasks, load_data
from models import GCNEncoder, HeavyPrompt, MAML, MLP_subgraph
from torch_geometric.utils import subgraph, k_hop_subgraph, to_dense_adj
import argparse
import torch.nn.functional as F
import os
from utils import acc_f1_over_batches, eval
from copy import deepcopy
from sklearn.model_selection import KFold
import random
from torch_geometric.data import Data
import numpy as np
import pickle as pkl

seed = 42
seed_everything(seed)

def meta_test_maml(args, maml, lossfn, tasks):
    assert len(tasks) == 1

    # device = args.device
    for i, (support, query) in enumerate(tasks):

        test_model = deepcopy(maml.module)
        test_opi =  optim.Adam(filter(lambda p: p.requires_grad, test_model.parameters()),
                              lr=0.0001,
                              weight_decay=0.001)
        test_model.train()
        
        support_loader = DataLoader(support, batch_size=args.batch_size, shuffle=False)
        query_loader = DataLoader(query, batch_size=args.batch_size, shuffle=False)

        for j in range(1):  # adaptation_steps
            running_loss = 0.
            for batch_id, xdata in enumerate(support_loader):
                # support_batch_x = support_batch_x.to(device)
                # support_batch_y = support_batch_y.to(device)
                data = Data(x = xdata['x'], adj=xdata['adj'], adj_norm=xdata['adj_norm'], labels = xdata['y'])
                data.to(args.device)

                support_batch_preds, mi_loss, gcl_loss = test_model(data)
                support_batch_loss = lossfn(support_batch_preds, data.labels)+ args.mi_weight * mi_loss + gcl_loss
                test_opi.zero_grad()
                support_batch_loss.backward()
                test_opi.step()
                running_loss += support_batch_loss.item()
                if batch_id == len(support_loader) - 1:  # report every PrintN updates
                    last_loss = running_loss / len(support_loader)   # loss per batch
                    print('training loss: {:.4f}'.format(last_loss))
                    running_loss = 0.
        test_model.eval()
        # acc_f1_over_batches(query_loader, test_model, device)
        micro, auc, macro = eval(test_model, query_loader, args)       
    return macro, micro, auc
    

def meta_train_maml(args, maml, lossfn, opt, tasks):
    # device = args.device
    learner = maml
    PrintN = 10
    losses = []
    for ep in range(args.epochs):
        meta_train_loss = 0.0
    
        for i, (support, query) in enumerate(tasks):
            
            support_loader = DataLoader(support, batch_size=args.batch_size, shuffle=True)
            query_loader = DataLoader(query, batch_size=args.batch_size, shuffle=True)

            for j in range(args.adapt_steps):  # adaptation_steps
                running_loss = 0.
                support_loss = 0.
                for batch_id, xdata in enumerate(support_loader):
                    # support_batch_x = support_batch_x.to(device)
                    # support_batch_y = support_batch_y.to(device)
                    data = Data(x = xdata['x'], adj=xdata['adj'], adj_norm=xdata['adj_norm'], labels = xdata['y'])
                    data.to(args.device)

                    support_batch_preds, mi_loss, gcl_loss = learner(data)
                    support_batch_loss = lossfn(support_batch_preds, data.labels)+ gcl_loss + args.mi_weight * mi_loss
                    # learner.adapt(support_batch_loss)
                    if not args.meta_test:
                        print(f'loss: {support_batch_loss:.4f}, mi_loss: {mi_loss:.4f}, gcl_loss: {gcl_loss:.4f}')
                    running_loss += support_batch_loss.item()
                    support_loss += support_batch_loss
                    if (batch_id + 1) % PrintN == 0:  # report every PrintN updates
                        last_loss = running_loss / PrintN  # loss per batch
                        print('task {}, adapt {}/{} | batch {}/{} | loss: {:.4f}'.format(i+1, j + 1, args.adapt_steps,
                                                                                batch_id + 1,
                                                                                len(support_loader),
                                                                                last_loss))

                        running_loss = 0.

                support_loss = support_loss / len(support_loader)
                learner.adapt(support_loss)

            running_loss, query_loss = 0., 0.
            for batch_id, xdata in enumerate(query_loader):
                # query_batch_x = query_batch_x.to(device)
                # query_batch_y = query_batch_y.to(device)
                data = Data(x = xdata['x'], adj=xdata['adj'], adj_norm=xdata['adj_norm'], labels = xdata['y'])
                data.to(args.device)
                
                query_batch_preds, mi_loss, gcl_loss = learner(data)
                query_batch_loss = lossfn(query_batch_preds, data.labels) + args.mi_weight * mi_loss + gcl_loss
                query_loss += query_batch_loss
                running_loss += query_batch_loss
                if (batch_id + 1) % PrintN == 0:
                    last_loss = running_loss / PrintN
                    print('task {}, query loss batch {}/{} | loss: {:.4f}'.format(i+1, batch_id + 1,
                                                                         len(query_loader),
                                                                         last_loss))

                    running_loss = 0.

            query_loss = query_loss / len(support_loader)
            meta_train_loss += query_loss

        print('meta_train_loss @ epoch {}/{}: {}'.format(ep+1, args.epochs, meta_train_loss.item()))
        meta_train_loss = meta_train_loss / len(tasks)
        opt.zero_grad()
        meta_train_loss.backward()
        opt.step()
        losses.append(meta_train_loss)
    with open('meta_train_loss.pkl', 'wb') as f:
        pkl.dump(losses, f)

class Discriminator(torch.nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.args = args
        self.input_size = self.args.hid_dim*2
        self.hidden_size = self.args.hid_dim
        self.fc1 = torch.nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()
        #torch.nn.init.constant(self.fc1.weight, 0.01)
        #torch.nn.init.constant(self.fc2.weight, 0.01)
    def forward(self, embeddings,positive):
        cat_embeddings = torch.cat((embeddings, positive),dim = -1)
        pre = self.relu(self.fc1(cat_embeddings))
        pre = torch.sigmoid(self.fc2(pre))
        return pre

class PromptModel(nn.Module):
    def __init__(self, args):
        super(PromptModel, self).__init__()
        self.batch_size=args.batch_size
        self.k = args.top_k
        #self.meta_train = not args.meta_test
        self.fc_in = nn.Linear(args.inp_dim, args.hid_dim)
        self.linear_out = nn.Linear(args.hid_dim, args.out_dim)
        self.linear_out1 = nn.Linear(args.hid_dim*3, args.out_dim)

        self.discriminator = Discriminator(args)
        self.gcn = GCNEncoder(nfeat=args.hid_dim, nhid=args.hid_dim, dropout=args.dropout)
        self.gcn_sup = GCNEncoder(nfeat=args.hid_dim, nhid=args.hid_dim, dropout=args.dropout, graph=True)
        self.gib = MLP_subgraph(inp_dim=args.hid_dim, hid_dim=args.hid_dim*2, num_nodes=args.num_nodes, tau=args.gumbel_tau)
        #self.mse_criterion = torch.nn.MSELoss(reduction='mean')
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def sample_sub_graph(self, adj, impt_idx, node_score):
        """
        Args:
            adj: (B, N, N)
            impt_idx: (B, K)
            node_score: (B, N)
        Return:
            (B,N,N)
        """
        subgraphs = []
        batch_size = adj.shape[0]
        for i in range(batch_size):
            seed_node = impt_idx[i]
            num_nodes = adj[i].shape[0]
            edge_index = torch.nonzero(adj[i]).t().contiguous()
            _, sub_edge_index, _, _ = k_hop_subgraph(node_idx=seed_node, num_hops=1, num_nodes=num_nodes, edge_index=edge_index, relabel_nodes=False)
            sup_mask = to_dense_adj(sub_edge_index, max_num_nodes=num_nodes)[0]
            # add important edges
            sup_adj  = adj[i] * sup_mask
            # import pdb; pdb.set_trace()
            for j in range(len(seed_node)):
                for k in range(j+1, len(seed_node)):
                    j_ = seed_node[j]
                    k_ = seed_node[k]
                    sup_adj[j_,k_] = sup_adj[k_,j_] = (node_score[i][j_]+node_score[i][k_])/2
            subgraphs.append(sup_adj)
        subgraphs = torch.stack(subgraphs)
        return subgraphs

    def MI_Est(self, discriminator, embeddings, positive):
        batch_size = embeddings.shape[0]
        shuffle_embeddings = embeddings[torch.randperm(batch_size)]
        joint = discriminator(embeddings,positive)
        margin = discriminator(shuffle_embeddings,positive)
        mi_est = torch.mean(joint) - torch.log(torch.mean(torch.exp(margin)))
        return mi_est

    def supervise_classify_loss(self,embeddings,positive,labels):
        data = torch.cat((embeddings, positive), dim=0)
        labels = torch.cat((labels, labels),dim=0)
        #if torch.cuda.is_available():
        #    labels = labels.cuda()
        pred = self.linear_out(data)
        loss = self.criterion(pred, labels)
        return loss

    def prompt_gcn(self, x, adj):
        """
        Args:
            x: (batch_size, node_num, d)
            adj: (batch_size, node_num, node_num)
            pg_x: (token_num, d)
            pg_adj: (token_num, token_num)
            cross_adj: (batch_size, token_num, nonde_num)
        """
        x = self.gcn(x, adj)
        return x, adj

    def forward(self, data):
        """
        Args:
            data: Data object
        """
        out=[]
        x = self.fc_in(data.x) 
        x1, _ = self.prompt_gcn(x, data.adj_norm)
        g_h=torch.mean(x1, dim=1)
        out.append(g_h)

        #node_score = node_score_batch(x1) #(B,N)
        node_score = entropy_node_score(x1, data.adj_norm) #(B,N)
        _, indx = torch.topk(node_score, k=self.k,largest = False) # (B, k) 
        sup_graph = self.sample_sub_graph(data.adj, indx, node_score)
        #gib_mask, conloss = self.gib(x1, pg)
        gib_mask = self.gib(x1)
        sup_graph_gib = sup_graph * gib_mask
        g_hid2 = self.gcn_sup(x1, sup_graph_gib)

        out.append(g_hid2)
        positive = self.gcn_sup(x1, sup_graph)
        out.append(positive)
        #mi_loss = gib_loss(g_hid2, g_hid1)
        mi_loss = self.MI_Est(self.discriminator, g_hid2, positive)
        # mi_loss = 0
        gcl_loss = self.supervise_classify_loss(g_hid2, positive, data.labels)
        #gcl_loss = gcl_loss_f(g_hid2, torch.mean(x1, dim=1))

        out_all = torch.cat(out, dim=-1)
        output = self.linear_out1(out_all)
        return output, mi_loss, gcl_loss

def model_components(args):
    adapt_lr = 0.001
    meta_lr = 0.001

    model = PromptModel(args).to(args.device)
    maml = MAML(model, lr=adapt_lr)

    opt = optim.Adam(filter(lambda p: p.requires_grad, maml.parameters()), meta_lr)
    lossfn = nn.CrossEntropyLoss()

    return maml, opt, lossfn



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--inp_dim', type=int,default=None)
    arg_parser.add_argument('--hid_dim', type=int, default=128)
    arg_parser.add_argument('--out_dim', type=int, default=2)
    arg_parser.add_argument('--token_num', type=int, default=10)
    arg_parser.add_argument('--token_dim', type=int, default=128)
    arg_parser.add_argument('--cross_prune', type=float, default=0.1)
    arg_parser.add_argument('--inner_prune', type=float, default=0.3)
    arg_parser.add_argument('--top_k', type=int, default=5)
    arg_parser.add_argument('--epochs', type=int, default=50)
    arg_parser.add_argument('--dataset', default='PPMI')
    arg_parser.add_argument('--seed', type=int, default=42)
    arg_parser.add_argument('--adapt_steps', type=int, default=1)
    arg_parser.add_argument('--mi_weight',type=float,default=1.)
    arg_parser.add_argument('--gpu', default="0")
    arg_parser.add_argument('--dropout', type=float, default=0.05)
    arg_parser.add_argument('--gumbel_tau', type=float, default=0.4)
    arg_parser.add_argument('--batch_size', type=int, default=16)
    arg_parser.add_argument('--meta_test', action='store_true')

    args = arg_parser.parse_args()
    args.device = torch.device("cuda:"+args.gpu)
    if args.dataset == 'PPMI':
        args.num_nodes = 84
    elif args.dataset == 'HIV':
        args.num_nodes = 90
    if args.inp_dim is None:
        args.inp_dim = args.num_nodes

    maml, opt, lossfn = model_components(args)

    # meta training on source tasks
    if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
    if not args.meta_test:
        tasks = load_tasks(args.dataset, seed=seed, n_task=3)
        print(len(tasks), "tasks loaded")
        meta_train_maml(args, maml, lossfn, opt, tasks)
        torch.save(maml.state_dict(), 'checkpoints/metapre_{}.pt'.format(args.dataset))
    else:
        # tasks = load_tasks(args.dataset, seed, n_task=4)
        #tasks_train = tasks[:3]
        #tasks_test = tasks[3:]
        # print(len(tasks), "tasks loaded")

        for name, param in maml.module.named_parameters():
            print(name, param.shape)
        # import pdb; pdb.set_trace()
        ckpt = torch.load('checkpoints/metapre_PPMI.pt',map_location=args.device)
        for name, param in maml.module.named_parameters():
            if not name.startswith("fc_in") and not name.startswith("gib.linear2"):
                assert param.shape == ckpt['module.'+name].shape, f"{name}, {param.shape}!={ckpt['module.'+name].shape}"
                param.data = ckpt['module.'+name]
        # maml.load_state_dict(torch.load('checkpoints/metapre_PPMI.pt'))
        # for name, param in maml.module.named_parameters():
        #     if  name.startswith('fc_in') or name.startswith("gib.linear2"):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        #meta_train_maml(args, maml, lossfn, opt, tasks_train)
        #torch.save(maml.state_dict(), 'checkpoints/metatest_{}.pt'.format(args.dataset))
        # maml.load_state_dict(torch.load('checkpoints/metatest_{}.pt'.format(args.dataset),map_location=args.device))
        print('Test')
        # meta_test_maml(args, maml, lossfn, tasks)
        data_path = os.path.join('data', args.dataset, args.dataset+'.mat')
        data, labels = load_data(data_path)
        print("data size", len(data))
        # random.seed(args.seed)
        accs = []
        aucs = []
        macros = []
        kf = KFold(n_splits=5, random_state=args.seed, shuffle=True)
        for k, (train_index, test_index) in enumerate(kf.split(data, labels)):
            train_set = [data[i] for i in train_index]
            test_set = [data[i] for i in test_index]
            train_task = load_tasks(data=train_set)
            test_task = load_tasks(data=test_set)

            meta_train_maml(args, maml, lossfn,opt, train_task)
            torch.save(maml.state_dict(), 'checkpoints/metatest_{}.pt'.format(args.dataset))
            
            # maml.load_state_dict(torch.load('checkpoints/metatest_{}.pt'.format(args.dataset),map_location=args.device))
            test_macro, test_micro, test_auc = meta_test_maml(args, maml, lossfn, test_task)
            # train_loader = DataLoader(pyDataset(train_set), batch_size=args.batch_size, shuffle=False)
            # test_loader = DataLoader(pyDataset(test_set), batch_size=args.batch_size, shuffle=False)
            # model = metapre.PromptModel(args).to(args.device)
            # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            # test_micro, test_auc, test_macro = train_and_evaluate(model, train_loader, test_loader,
                                                                                # optimizer,args)
            print(f'Fold {k}, (Initial Performance Last Epoch) | test_micro={(test_micro * 100):.2f}, '
                            f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}')
            accs.append(test_micro)
            aucs.append(test_auc)
            macros.append(test_macro)
        mean_acc = np.mean(np.array(accs))
        std_acc = np.std(np.array(accs))
        mean_auc = np.mean(np.array(aucs))
        std_auc = np.std(np.array(aucs))
        mean_mac = np.mean(np.array(macros))
        std_mac = np.std(np.array(macros))
        print("Mean Acc:", mean_acc,'±',std_acc)
        print("Mean Auc:", mean_auc,'±',std_auc)
        print("Mean Mac:", mean_mac,'±',std_mac)  
