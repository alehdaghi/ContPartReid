import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd


class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output


def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et


def learn_mine(joint, marginal, mine_net, ma_et=1.0, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    device = joint.device
    # joint = torch.autograd.Variable(torch.FloatTensor(joint)).to(device)
    # marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).to(device)
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)

    # unbiasing use moving average
    # loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
    # use biased estimator
    loss = -mi_lb

    return loss, ma_et

def sample_batch(feats1, feats2):
    joint = torch.cat([feats1, feats2], dim=1)
    marginal_index = torch.randperm(feats1.size(0))
    feats2_p = feats2[marginal_index]
    marginal = torch.cat([feats1, feats2], dim=1)
    return joint, marginal

def estimate_MI(feats1, feats2, mine_net):
    joint, marginal = sample_batch(feats1, feats2)
    return learn_mine(joint, marginal, mine_net)[0]