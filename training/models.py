import torch
from torch import nn


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

        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

        if weights:
            self.out_proj = nn.Linear(dim, input_features)

            # self.alpha = nn.Parameter(torch.rand(1, 2, 1))
            # torch.nn.init.kaiming_normal_(self.alpha)

            self.first_idx = torch.tensor([0, 1, 2, 6, 7, 10, 11], device="cuda")
            self.second_idx = torch.tensor([3, 4, 5, 8, 9, 12, 13], device="cuda")
        else:
            self.out_proj = nn.Linear(dim, 2)

        self.activation = get_activation(activation)

    def _apply_weighted_sum(
        self, outputs: torch.Tensor, feats: torch.Tensor
    ) -> torch.Tensor:
        B, F = outputs.shape

        feats1 = feats.index_select(dim=1, index=self.first_idx)  # [B, F1]
        feats2 = feats.index_select(dim=1, index=self.second_idx)  # [B, F2]
        feats_stacked = torch.stack([feats1, feats2], dim=1)  # [B, 2, F_group]

        w = outputs.view(B, 2, F // 2)  # [B, 2, F/2]
        w = torch.softmax(w, dim=-1)  # [B, 2, F/2]
        # w = w / torch.sum(w, dim=-1, keepdim=True)

        # alpha = 1.5  # can be >1 to allow values >1 and <0
        # beta = (1 - self.alpha) / (F // 2)
        # w = self.alpha * w + beta

        return torch.einsum("bof,bof->bo", w, feats_stacked)  # [B, 2]

    def forward(self, data):
        x = data["feats"]  # [B, F]
        in_x = x.clone()

        x = self.activation(self.in_proj(x))

        for layer in self.layers:
            x = self.activation(layer(x))

        x = self.out_proj(x)

        if self.weights:
            x = self._apply_weighted_sum(x, in_x)

        out_dict = {"preds": x}

        return out_dict

    @torch.no_grad()
    def generate(self, x):
        in_x = x.clone()
        x = self.activation(self.in_proj(x))

        for layer in self.layers:
            x = self.activation(layer(x))

        x = self.out_proj(x)

        if self.weights:
            x = self._apply_weighted_sum(x, in_x)

        return x

    def loss(self, pred, data):
        pred_invs = pred["preds"]
        target_invs = data["targets"]

        mse_loss = nn.functional.mse_loss(pred_invs, target_invs, reduction="none")

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
    model.to("cuda")

    return model
