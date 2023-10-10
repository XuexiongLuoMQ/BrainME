import torch
import torch.nn as nn


class MAML(nn.Module):
    def __init__(self, model, lr):
        super(MAML, self).__init__()
        self.module = model
        self.lr = lr

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    @staticmethod
    def _Adam(param, alpha, grad):
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-8
        m_t = 0
        v_t = 0
        t = 0
        for _ in range(1):
            t += 1
            m_t = beta_1 * m_t + (1 - beta_1) * grad
            v_t = beta_2 * v_t + (1 - beta_2) * (torch.pow(grad, 2))
            m_cap = m_t / (1 - (beta_1 ** t))
            v_cap = v_t / (1 - (beta_2 ** t))
            param = param - (alpha * m_cap) / (torch.sqrt(v_cap) + epsilon)
        return param

    def adapt(self, loss): 
        params = [p for p in self.module.parameters() if p.requires_grad]
        # with torch.autograd.detect_anomaly():
        gradients = torch.autograd.grad(loss, params)
        for param, grad in zip(params, gradients):
            param.data = self._Adam(param, self.lr, grad)