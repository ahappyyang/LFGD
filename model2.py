import torch
import torch.nn as nn
import torch.nn.functional as F
from GCN import GCN
from torch.nn.parameter import Parameter


class my_model(nn.Module):
    def __init__(self, dims,img_size,num_classes):
        super(my_model, self).__init__()
        self.layers1 = nn.Linear(dims[0], dims[1])
        self.layers2 = nn.Linear(dims[0], dims[1])
        self.layers2 = nn.Linear(dims[0], dims[1])
        self.layer_GCN = nn.Sequential()

        for i in range(2):
            self.layer_GCN.add_module('GCN_Branch' + str(i), GCN(height=img_size, width=img_size, changel= dims[0], layers_count=1))

        self.dropout = nn.Dropout(p=0.4)

        self.cluster_layer = Parameter(torch.Tensor(num_classes, dims[1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = 1

    def forward(self, x, emb, is_train=True, sigma=0.01):
        a = torch.mm(emb, emb.t())
        A = F.normalize(a, dim=1, p=1)
        x= x.unsqueeze(0)
        A = A.unsqueeze(0)
        for i in range(len(self.layer_GCN)):
            x = self.layer_GCN[i](x, A)
            x = self.dropout(x)

        x = x.squeeze(0)


        out1 = self.layers1(x)
        out2 = self.layers2(x)

        out1 = F.normalize(out1, dim=1, p=2)

        if is_train:
            out2 = F.normalize(out2, dim=1, p=2) + torch.normal(0, torch.ones_like(out2) * sigma).cuda()
            # out2 = F.normalize(out2, dim=1, p=2)
        else:
            out2 = F.normalize(out2, dim=1, p=2)

        q = self.get_Q(out2)
        p = self.target_distribution(q.detach())
        return out1, out2, q, p

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def target_distribution(self,q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

# class my_model(nn.Module):
#     def __init__(self, dims,img_size):
#         super(my_model, self).__init__()
#         self.layers1 = nn.Linear(dims[0], dims[1])
#         self.layers2 = nn.Linear(dims[0], dims[1])
#         self.layers2 = nn.Linear(dims[0], dims[1])
#         self.layer_GCN = nn.Sequential()
#
#         for i in range(5):
#             self.layer_GCN.add_module('GCN_Branch' + str(i), GCN(height=img_size, width=img_size, changel= dims[0], layers_count=1))
#
#         self.dropout = nn.Dropout(p=0.3)
#
#     def forward(self, x, emb, is_train=True, sigma=0.01):
#         a = torch.mm(emb, emb.t())
#         A = F.normalize(a, dim=1, p=1)
#         # A= a
#         x= x.unsqueeze(0)
#         A = A.unsqueeze(0)
#         for i in range(len(self.layer_GCN)):
#             x = self.layer_GCN[i](x, A)
#             x = self.dropout(x)
#
#         x = x.squeeze(0)
#
#
#         out1 = self.layers1(x)
#         out2 = self.layers2(x)
#
#         out1 = F.normalize(out1, dim=1, p=2)
#
#         if is_train:
#             # out2 = F.normalize(out2, dim=1, p=2) + torch.normal(0, torch.ones_like(out2) * sigma).cuda()
#             out2 = F.normalize(out2, dim=1, p=2)
#         else:
#             out2 = F.normalize(out2, dim=1, p=2)
#         return out1, out2

# class my_model(nn.Module):
#     def __init__(self, dims):
#         super(my_model, self).__init__()
#         self.layers1 = nn.Linear(dims[0], dims[1])
#         self.layers2 = nn.Linear(dims[0], dims[1])
#         self.layers2 = nn.Linear(dims[0], dims[1])
#
#     def forward(self, x, is_train=True, sigma=0.01):
#         out1 = self.layers1(x)
#         out2 = self.layers2(x)
#
#         out1 = F.normalize(out1, dim=1, p=2)
#
#         if is_train:
#             out2 = F.normalize(out2, dim=1, p=2) + torch.normal(0, torch.ones_like(out2) * sigma).cuda()
#         else:
#             out2 = F.normalize(out2, dim=1, p=2)
#         return out1, out2