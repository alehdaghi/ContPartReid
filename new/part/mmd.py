import numpy as np
import torch
import torch.nn as nn

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


def partMMD(t, q_feats, g_feats , q_ids, g_ids):
    mmd = MMD_loss()
    ids = np.unique(q_ids)
    S_Z, T_Z, S_T, Z_Z = 0, 0, 0, 0
    # p = q_feats.shape[1]
    for id in ids:
        Q, G = q_feats[q_ids == id], g_feats[g_ids == id]
        l = min(Q.shape[0], G.shape[0])
        S, T = Q[:l], G[:l]
        Ss = S.reshape(l, -1, 2048)
        Tt = T.reshape(l, -1, 2048)
        Zs, Zt = step(t, Ss, Tt)
        S_Z = S_Z + mmd(Ss.reshape(l, -1), Zs.reshape(l, -1))
        T_Z = T_Z + mmd(Tt.reshape(l, -1), Zt.reshape(l, -1))
        S_T = S_T + mmd(Ss.reshape(l, -1), Tt.reshape(l, -1))
        Z_Z = Z_Z + mmd(Zs.reshape(l, -1), Zt.reshape(l, -1))
    return S_Z / len(ids), T_Z / len(ids), S_T / len(ids), Z_Z / len(ids)

def step(t, v, i):
    part_num = i.shape[1]
    if t == 0:
        f1 = v
        f2 = i
    elif t < part_num:
        f1 = v.clone()
        f2 = i.clone()
        for j in range(v.shape[0]):
            index1 = np.random.choice(part_num, t, False)
            index2 = np.random.choice(part_num, t, False)
            f1[j][index1] = i[j][index1]
            f2[j][index2] = v[j][index2]
    else:
        f1 = i
        f2 = v
    return f1 , f2
