import torch
from torch import nn


# class FEDModel(nn.Module):
#     def __init__(self, input_dim=100, hidden_dim=256):#default:input_dim=200
#
#         super(FEDModel, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.internal_dim = int(hidden_dim / 2)
#         self.loss_function = nn.NLLLoss(reduction='none')
#
#         self.layer1 = nn.Sequential(
#             nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim, bias=True),
#             nn.ReLU(),
#             nn.Dropout(p=0.2)
#         )
#         self.feature = nn.ModuleList([nn.Sequential(
#             nn.Linear(in_features=self.hidden_dim, out_features=self.internal_dim, bias=True),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(in_features=self.internal_dim, out_features=self.hidden_dim, bias=True),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#
#         ) for _ in range(1)])
#
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=self.hidden_dim, out_features=2),
#             nn.LogSoftmax(dim=-1)
#         )
#
#
#     def extract_feature(self, x):
#         out = self.layer1(x)
#         for layer in self.feature:
#             out = layer(out)
#
#         return out
#
#     def forward(self, example_batch,targets=None):
#         h_a = self.extract_feature(example_batch)
#         y_a = self.classifier(h_a)
#         probs = torch.exp(y_a)
#         if targets is not None:
#             ce_loss = self.loss_function(input=y_a, target=targets)
#             batch_loss = (ce_loss).sum(dim=-1)
#             return probs, h_a, batch_loss
#         else:
#             return probs, h_a
#         pass
import torch
import torch.nn as nn


class FEDModel(nn.Module):
    def __init__(self, input_dim=171, hidden_dim=256):  # 修改 input_dim 为 171

        super(FEDModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.internal_dim = int(hidden_dim / 2)
        self.loss_function = nn.NLLLoss(reduction='none')

        # layer1 输入维度已修改为 self.input_dim = 171
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.feature = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.internal_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.internal_dim, out_features=self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        ) for _ in range(1)])

        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=self.hidden_dim, out_features=2),
        #     nn.LogSoftmax(dim=-1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=3),  # 将输出类别数改为 3
            nn.LogSoftmax(dim=-1)
        )



    def extract_feature(self, x):
        out = self.layer1(x)
        for layer in self.feature:
            out = layer(out)
        return out

    def forward(self, example_batch, targets=None):
        h_a = self.extract_feature(example_batch)
        y_a = self.classifier(h_a)
        probs = torch.exp(y_a)

        if targets is not None:
            ce_loss = self.loss_function(input=y_a, target=targets)
            batch_loss = (ce_loss).sum(dim=-1)
            return probs, h_a, batch_loss
        else:
            return probs, h_a
