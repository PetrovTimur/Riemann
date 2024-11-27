import numpy as np

g = 9.8066


h_k = np.array([5, 4])
u_k = np.array([10, 10])
c_k = np.sqrt(g * h_k)

c = 0.25 * (u_k[0] - u_k[1]) + 0.5 * c_k.sum()

grad = 1
eps = 1e-15

phi_k = np.array([0, 0])
dphi_k = np.array([0, 0])


k = 0
while abs(grad) > eps:
    k += 1
    s_k = c / c_k
    phi_k = np.where(s_k >= 1, ((c - c_k) * (s_k + 1) * (1 + s_k ** -2) ** 0.5 / (2 ** 0.5)), 2 * (c - c_k))
    dphi_k = np.where(s_k >= 1, ((2 * s_k ** 2 + 1 + s_k ** -2) / (2 ** 0.5 ** s_k ** (1 + s_k ** -2) ** 0.5)), 2)
    c -= grad

    grad = (phi_k.sum() - (u_k[0] - u_k[1])) / (dphi_k.sum())
    print(f'k = {k}, F = {phi_k}, grad = {grad}')

print(c)

h = c ** 2 / g
u = 0.5 * (u_k.sum() + phi_k[1] - phi_k[0])
print(f'h = {h}, u = {u}')