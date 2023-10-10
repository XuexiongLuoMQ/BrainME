import random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
import torchmetrics
from sklearn import metrics
from torch_geometric.data import Data


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def node_score_batch(emb, k_near=4):
    """
    Args:
        emb: (B, N, d)
    Return:
        node score (B, N)
    """
    emb = emb.data
    B = emb.shape[0]
    sim = torch.matmul(emb, emb.transpose(1,2))
    _, I = torch.topk(sim, k_near)
    k_emb = []
    for i in range(B):
        k_emb.append(emb[i, I[i]])
    k_emb = torch.stack(k_emb)  # (B,N,K,d)
    emb = k_emb.sum(-2)
    emb_w = F.normalize(emb, dim=-1)
    emb = emb * emb_w
    emb = F.relu(emb)
    score = emb.sum(-1)
    score = score / score.max(1, keepdim=True)[0]
    return score


def entropy_node_score(emb, adj, alpha=1.):
    """
    Args:
        emb: (B, N, d)
        adj: (B, N, N), add self-loop adj
    """
    B, N, _ = adj.shape
    adj = (adj != 0).float()
    s = torch.zeros(B, N)

    # import pdb; pdb.set_trace()
    emb_g = torch.mean(emb, dim=1, keepdim=True)  # (B,1,d)
    q = (emb * emb_g).sum(dim=2)  # (B,N)
    # s1 = torch.exp(q)  # OVERFLOW
    s1 = torch.sigmoid(q)
    s1 = s1 / s1.sum(dim=1, keepdim=True)
    d = adj.sum(dim=2)  

    for i in range(B):
        inds = torch.nonzero(adj[i])
        for j in range(N):    
            neigh = inds[inds[:,0]==j]
            d_neigh = d[i, neigh[:,1]]
            p = d_neigh / d_neigh.sum()
            emb_neigh = s1[i, neigh[:,1]]
            p1 = emb_neigh / emb_neigh.sum()
            s[i,j] = torch.sum(-p * torch.log(p)) + alpha*torch.sum(-p1 * torch.log(p1))


    return s








def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    dist= pairwise_distances(x)
    k = torch.exp(-dist /sigma)
    k = k/torch.trace(k)
    return k

def reyi_entropy(x,sigma):
    alpha = 1.01
    k = calculate_gram_mat(x,sigma)
    # k = k/torch.trace(k) 
    # eigv = torch.abs(torch.symeig(k, eigenvectors=False)[0])
    # eig_pow = eigv**alpha
    x = torch.trace(k**alpha)
    entropy = (1/(1-alpha))*torch.log2(x)
    return entropy


def joint_entropy(x,y,s_x,s_y):
    alpha = 1.01
    x = calculate_gram_mat(x,s_x)
    y = calculate_gram_mat(y,s_y)
    k = torch.mul(x,y)
    k = k/torch.trace(k)
    # eigv = torch.abs(torch.symeig(k, eigenvectors=False)[0])
    # eig_pow =  eigv**alpha
    x = torch.trace(k**alpha)
    entropy = (1/(1-alpha))*torch.log2(x)

    return entropy

# calculate mutual information
def calculate_MI(x,y,s_x,s_y):
    
    Hx = reyi_entropy(x,sigma=s_x)
    Hy = reyi_entropy(y,sigma=s_y)
    Hxy = joint_entropy(x,y,s_x,s_y)
    Ixy = Hx+Hy-Hxy
    
    return Ixy



def gib_loss(positive, embeddings):
    # calculate to sigma1 and sigma2
    with torch.no_grad():
        Z_numpy1 = embeddings.cpu().detach().numpy()
        k = squareform(pdist(Z_numpy1, 'euclidean'))
        k = k[~np.eye(k.shape[0], dtype=bool)].reshape(k.shape[0], -1)
        sigma1 = np.mean(np.sort(k, 1)[:,:10])

    with torch.no_grad():
        Z_numpy2 = positive.cpu().detach().numpy()
        k = squareform(pdist(Z_numpy2, 'euclidean'))
        k = k[~np.eye(k.shape[0], dtype=bool)].reshape(k.shape[0], -1)
        sigma2 = np.mean(np.sort(k, 1)[:,:10])

    mi_loss = calculate_MI(embeddings, positive, sigma1**2, sigma2**2)

    # mi_loss = calculate_MI(embeddings,positive, sigma2**2,sigma1**2)
    return mi_loss


def acc_f1_over_batches(test_loader, model, device, num_class=2, task_type='multi_class_classification'):
    if task_type == "multi_class_classification":
        accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class)
        macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro")
    elif task_type == "binary_classification":
        accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class)
        macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro")
    else:
        raise NotImplementedError

    accuracy = accuracy.to(device)
    macro_f1 = macro_f1.to(device)

    for batch_id, (test_batch_x, test_batch_y) in enumerate(test_loader):
        test_batch_x = test_batch_x.to(device)
        test_batch_y = test_batch_y.to(device)

        pre, _, _ = model(test_batch_x)

        pre = pre.detach()
        y = test_batch_y

        pre_cla = torch.argmax(pre, dim=1)
        # print(pre_cla)
        # print(y)

        # import pdb; pdb.set_trace()
        acc = accuracy(pre_cla, y)
        ma_f1 = macro_f1(pre_cla, y)
        print("Batch {} Acc: {:.4f} | Macro-F1: {:.4f}".format(batch_id, acc.item(), ma_f1.item()))

    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    print("Final True Acc: {:.4f} | Macro-F1: {:.4f}".format(acc.item(), ma_f1.item()))
    accuracy.reset()
    macro_f1.reset()


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
        #print(c,'33333333333')
        pred = c.max(dim=1)[1]
        preds += pred.detach().cpu().tolist()
        preds_prob += c[:, 1].detach().cpu().tolist()
        trues += data.labels.detach().cpu().tolist()
    fpr, tpr, _ = metrics.roc_curve(trues, preds_prob)
    train_auc = metrics.auc(fpr, tpr)
    if np.isnan(train_auc):
        train_auc = 0.5
    train_micro = metrics.f1_score(trues, preds, average='micro')
    train_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])
        
    return train_micro, train_auc, train_macro

if __name__ == '__main__':
    data = torch.load('x_adj.debug.pt')
    node_score = entropy_node_score(data['x'], data['adj'])
    print(torch.isnan(node_score).any())