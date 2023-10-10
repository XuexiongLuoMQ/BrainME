# import faiss
import torch
import torch.nn.functional as F


def normalize(x, dim):
    x = x / torch.linalg.norm(x, dim=dim, keepdims=True)
    return x

def k_near_select(emb, k_near):
    """
    Args:
        emb: (b, N, d)
    """
    emb1 = emb.data
    b = emb1.shape[0]
    # index = faiss.IndexFlatL2(emb1.shape[1])  # select via L2 distance 
    # index.add(emb1) 
    # _, I = index.search(emb1, k_near)
    sim = torch.matmul(emb1, emb1.transpose(1,2))
    _, I = torch.topk(sim, k_near)
    emb = emb[range(b), I, :].sum(-2)
    emb_w = normalize(emb, dim=-1)
    emb = emb * emb_w
    emb = F.relu(emb)
    emb = emb.sum(-1)

    return emb