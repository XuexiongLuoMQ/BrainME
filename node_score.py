# import faiss
import torch
import torch.nn.functional as F


def node_score_batch(emb, k_near):
    """
    Args:
        emb: (b, N, d)
    """
    emb = emb.data
    b = emb.shape[0]
    sim = torch.matmul(emb, emb.transpose(1,2))
    _, I = torch.topk(sim, k_near)
    emb = emb[range(b), I, :].sum(-2)
    emb_w = F.normalize(emb, dim=-1)
    emb = emb * emb_w
    emb = F.relu(emb)
    emb = emb.sum(-1)

    return emb