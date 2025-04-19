import torch
from torch import nn
from torch.nn.functional import elu

class MLP(nn.Module):
    def __init__(self, input_features):
        super(MLP, self).__init__()
        n_feats = 20
        self.layer1 = nn.Linear(input_features, n_feats)
        self.layer2 = nn.Linear(n_feats, n_feats)
        self.layer3 = nn.Linear(n_feats, n_feats)
        self.layer4 = nn.Linear(n_feats, n_feats)
        self.layer5 = nn.Linear(n_feats, 2)
        self.activation = elu
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x):
        # if x.dim() == 1:
        #     x = x.unsqueeze(1)
        # x = self.bn(x)
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.activation(self.layer4(x))
        x = self.layer5(x)
        return x

def load_nn(solver):
    model = None
    if solver.endswith('nn'):
        model = MLP(4)
        model.load_state_dict(torch.load('model.pt', weights_only=True))
        model.eval()
        model.to('cuda')

    return model