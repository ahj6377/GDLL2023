import torch
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
class Graph_Conv(nn.Module):
    def __init__(self, inp, out, bias=True):
        super(Graph_Conv, self).__init__()
        self.n_in  = inp
        self.n_out = out
        self.linear = nn.Linear(inp, out)

    def forward(self, x, adj):
        out = self.linear(x)
        return F.elu(torch.spmm(adj, out))


class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, layers, drop_out):
        super(GCN, self).__init__()
        self.layers = layers
        self.n_hid = n_hid
        self.gcs = nn.ModuleList()
        self.gcs.append(Graph_Conv(n_feat, n_hid))
        self.dropout = nn.Dropout(drop_out)
        for i in range(layers-1):
            self.gcs.append(Graph_Conv(n_hid, n_hid))

    def forward(self, x, adjs):
        for idx in range(len(self.gcs)):
            x = self.dropout(self.gcs[idx](x, adjs[idx]))
        return x

class GCNModel(nn.Module):
    def __init__(self, encd, num_classes, drop_out, inp):
        super(GCNModel, self).__init__()
        self.encd = encd
        self.dropout = nn.Dropout(drop_out)
        self.linear  = nn.Linear(self.encd.n_hid, num_classes)
    def forward(self, feat, adjs):
        x = self.encd(feat, adjs)
        x = self.dropout(x)
        x = self.linear(x)
        return x
