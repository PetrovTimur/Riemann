import torch
from torch import nn
from torch.nn.functional import elu


class MLP(nn.Module):
    def __init__(
        self,
        input_features: int = 4,
        n_layers: int = 4,
        dim: int = 20,
        activation: str = "elu",
        weights: bool = False,
    ):
        super(MLP, self).__init__()
        self.in_proj = nn.Linear(input_features, dim)

        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(n_layers)]
        )

        if weights:
            self.out_proj = nn.Linear(dim, 2 * input_features)
        else:
            self.out_proj = nn.Linear(dim, 2)

        self.activation = elu

    def forward(self, x):
        x = self.activation(self.in_proj(x))

        for layer in self.layers:
            x = self.activation(layer(x))

        x = self.out_proj(x)

        return x


def load_nn(path=None, input_features=4, n_feats=20):
    model = MLP(input_features=input_features, dim=n_feats)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    model.to('cuda')

    return model
