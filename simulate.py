import numpy as np
from solvers import GodunovSolver, CabaretSolver
import torch
from torch import nn
from torch.nn.functional import elu

def compute(h, hu, solver, dx, t_end):
    # print(h, hu)
    t = 0.0
    CFL = 0.3
    g = 9.81
    while t < t_end:
        u = np.where(h > 0, hu / h, 0)
        c = np.sqrt(g * h)
        dt = CFL * dx / np.max(np.abs(u) + c)
        h, hu = solver.step(h, hu, dx, dt)
        t += dt

    # print(t)
    # print(h)
    return h, hu

# Не используется
def insert_intermediate_points(h, hu):
    new_h = np.zeros((2 * h.shape[0] - 1))
    new_hu = np.zeros((2 * hu.shape[0] - 1))

    new_h[::2] = h
    # print(new_h.shape[0] // 2)
    new_h[new_h.shape[0] // 2] = (new_h[new_h.shape[0] // 2 - 2] + new_h[new_h.shape[0] // 2 + 2]) / 2
    new_h[1::2] = (h[:-1] + h[1:]) / 2


    # print(hu)
    new_hu[::2] = hu
    # print(hu[:-1], hu[1:])
    # print((hu[:-1] + hu[1:]) // 2)
    new_hu[new_hu.shape[0] // 2] = (new_hu[new_hu.shape[0] // 2 - 2] + new_hu[new_hu.shape[0] // 2 + 2]) / 2
    new_hu[1::2] = (hu[:-1] + hu[1:]) / 2


    return new_h, new_hu


def new_grid(h, hu):
    new_h = np.zeros(2 * len(h) + 1, dtype=np.float32)
    new_hu = np.zeros(2 * len(hu) + 1, dtype=np.float32)

    # Кладем начальные значения в консервативные точки (нечетные)
    new_h[1::2] = h
    new_hu[1::2] = hu

    # Считаем значения в потоковых (четных) точках (среднее консервативных)
    new_h[2:-1:2] = (h[1:] + h[:-1]) / 2
    new_hu[2:-1:2] = (hu[1:] + hu[:-1]) / 2

    # Граничные значения оставляем такими же
    new_h[0] = h[0]
    new_h[-1] = h[-1]
    new_hu[0] = hu[0]
    new_hu[-1] = hu[-1]

    return new_h, new_hu


def initial_conditions(nx, h_l, u_l, h_r, u_r):
    h = np.full(nx, h_r, dtype=np.float32)   # Изначально заполняем правым значением
    h[:nx // 2] = h_l   # В левую половину кладем левое значение

    u = np.full(nx, u_r, dtype=np.float32)
    u[:nx // 2] = u_l

    hu = h * u
    return h, hu

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

def run_simulation(L, nx, h_l, u_l, h_r, u_r, solver, t_end):
    dx = L / nx
    h0, hu0 = initial_conditions(nx, h_l, u_l, h_r, u_r)

    # print(h0, hu0)

    new_h, new_hu = new_grid(h0, hu0)
    # # print(new_h, new_hu)
    model = None
    if solver.endswith('nn'):
        model = MLP(4)
        model.load_state_dict(torch.load('model.pt', weights_only=True))
        model.eval()
        model.to('cuda')
    solver = CabaretSolver(solver_func=solver, model=model)
    h_final, hu_final = compute(new_h.copy(), new_hu.copy(), solver, dx, t_end)
    return h_final, hu_final

    # solver = GodunovSolver(solver_func='newton')
    # h_final, hu_final = compute(h0.copy(), hu0.copy(), solver, dx, t_end)
    # return h_final, hu_final

def error_norm(a, b, dx):
    return np.max(np.abs(a - b))
    # return np.sqrt(np.sum(dx * np.abs(a - b) ** 2))