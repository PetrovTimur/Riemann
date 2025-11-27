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
        self.weights = weights

        self.in_proj = nn.Linear(input_features, dim)

        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(n_layers)]
        )

        if weights:
            self.out_proj = nn.Linear(dim, 2 * input_features)
        else:
            self.out_proj = nn.Linear(dim, 2)

        self.activation = get_activation(activation)

    def forward(self, data):
        x = data["feats"]
        in_x = x.clone()

        x = self.activation(self.in_proj(x))

        for layer in self.layers:
            x = self.activation(layer(x))

        x = self.out_proj(x)

        if self.weights:
            w_out = x
            w = w_out.view(w_out.size(0), 2, 14)  # [B, 2, F]
            w = torch.softmax(w, dim=-1)  # softmax over features
            x = torch.einsum('bij,bj->bi', w, in_x)  # [B, 2]

        out_dict = {"preds": x}

        return out_dict
    
    def generate(self, x):
        in_x = x.clone()
        x = self.activation(self.in_proj(x))

        for layer in self.layers:
            x = self.activation(layer(x))

        x = self.out_proj(x)

        if self.weights:
            if in_x.ndim == 1:
                in_x = in_x.unsqueeze(0)
            w_out = x
            w = w_out.view(w_out.size(0), 2, 14)  # [B, 2, F]
            w = torch.softmax(w, dim=-1)  # softmax over features
            x = torch.einsum('bij,bj->bi', w, in_x)  # [B, 2]

        return x

    def loss(self, pred, data):
        pred_invs = pred["preds"]
        target_invs = data["targets"]

        mse_loss = nn.functional.mse_loss(pred_invs, target_invs)

        return {"mse_loss": mse_loss}



def get_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise NotImplementedError

def load_nn(path=None, input_features=4, n_feats=20):
    model = MLP(input_features=input_features, dim=n_feats)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    model.to('cuda')

    return model
