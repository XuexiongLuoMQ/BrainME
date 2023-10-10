from os import SEEK_END
import torch
import numpy as np
import process_data
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import metapre
import argparse
from sklearn import metrics

from torch_geometric.data import Data

import os
import random
import numpy
from process_data import pyDataset


def train_and_evaluate(model, train_loader, test_loader, optimizer, args):
    model.train()
    accs, aucs, macros = [], [], []
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(args.epoch_num):
        # print('epoch', i)
        loss_all = 0
        for xdata in train_loader:

            data = Data(x = xdata['x'], adj=xdata['adj'], adj_norm=xdata['adj_norm'], labels = xdata['y'])
            data.to(args.device)

        #for batch_id, （batch_x, batch_y）in enumerate(train_loader):
            optimizer.zero_grad()
            # print('forward')
            out, mi_loss, gcl_loss = model(data)
            # print('forward end')
            loss = criterion(out, data.labels) + gcl_loss + mi_loss  # MI LOSS results in NAN!!!
            loss.backward()
            optimizer.step()
            loss_all += loss.detach().cpu().item()
        epoch_loss = loss_all / len(train_loader.dataset)
        train_micro, train_auc, train_macro = eval(model,train_loader,args)
        title = "Start Train" 
        print(f'({title}) | Epoch={i:03d}, loss={epoch_loss:.4f}, \n'
                  f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                  f'train_auc={(train_auc * 100):.2f}')
        if (i + 1) % args.test_interval == 0:
            test_micro, test_auc, test_macro = eval(model, test_loader, args)
            accs.append(test_micro)
            aucs.append(test_auc)
            macros.append(test_macro)
            text = f'({title} Epoch {i}), test_micro={(test_micro * 100):.2f}, ' \
                       f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}\n'
            print(text)
            # with open(args.save_result, "a") as f:
            #         f.writelines(text)
    #torch.save(model,'GCN_pd.pkl')
    accs, aucs, macros = numpy.sort(numpy.array(accs)), numpy.sort(numpy.array(aucs)), \
                             numpy.sort(numpy.array(macros))
    return accs.max(), aucs.max(), macros.max()

def eval(model, loader, args):
    model.eval()
    preds, trues, preds_prob = [], [], []
    for xdata in loader:

        data = Data(x = xdata['x'], adj=xdata['adj'], adj_norm=xdata['adj_norm'], labels = xdata['y'])
        data.to(args.device)
        c,_,_=model(data)
        # if torch.isnan(c).any():
        #     import pdb; pdb.set_trace()
        c = torch.softmax(c,dim=1)
        pred = c.max(dim=1)[1]
        preds += pred.detach().cpu().tolist()
        preds_prob += c[:, 1].detach().cpu().tolist()
        trues += data.labels.detach().cpu().tolist()
    fpr, tpr, _ = metrics.roc_curve(trues, preds_prob)
    train_auc = metrics.auc(fpr, tpr)
    if numpy.isnan(train_auc):
        train_auc = 0.5
    train_micro = f1_score(trues, preds, average='micro')
    train_macro = f1_score(trues, preds, average='macro', labels=[0, 1])
        
    return train_micro, train_auc, train_macro


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HIV')
parser.add_argument('--inp_dim', type=int, default=90)
parser.add_argument('--num_nodes', type=int, default=90)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--token_dim', type=int, default=16)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--out_dim', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--gumbel_tau', type=float, default=0.4)
parser.add_argument('--seed', type=int, default=924)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--gpu', default="0")
parser.add_argument('--test_interval', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001) 

args = parser.parse_args()
args.device = torch.device("cuda:"+args.gpu)
data_path = os.path.join('data', args.dataset, args.dataset+'.mat')
data, labels = process_data.load_data(data_path)
print("data size", len(data))
random.seed(args.seed)
accs = []
aucs = []
macros = []
kf = KFold(n_splits=5, random_state=args.seed, shuffle=True)
for k, (train_index, test_index) in enumerate(kf.split(data, labels)):
    train_set = [data[i] for i in train_index]
    test_set = [data[i] for i in test_index]
    train_loader = DataLoader(pyDataset(train_set), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(pyDataset(test_set), batch_size=args.batch_size, shuffle=False)
    model = metapre.PromptModel(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    test_micro, test_auc, test_macro = train_and_evaluate(model, train_loader, test_loader,
                                                                           optimizer,args)
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


