import numpy as np
from fractions import Fraction

a = np.array([[8, -4, 0],
              [-4, 8, -4],
              [0, -4, 8]], dtype=np.float64)

f = np.array([3. / 8, 3. / 4, 9. / 8])    #1
# f = np.array([7 / 32, 25 / 32, 55 / 32], dtype=np.float64)    #2
# f = 1 / 128 * np.array([15, 90, 285], dtype=np.float64)    #3
# f = 1 / 128 * np.array([15, 90, 285], dtype=np.float64)    #4
# f = 2 * np.array([2, 1, 0], dtype=np.float64)    #5

# print(a)
# print(f)
#
inv = np.linalg.inv(a)
print(inv)
#
b = np.matmul(np.linalg.inv(a), f)
print(b)
#
for bb in b:
    print(Fraction(bb).limit_denominator(10000))
#
# print(21. / 64)

def integr(x_i, t):
    return 4 * x_i ** 3 - 3 * x_i ** 4 - 6 * t * x_i ** 2 + 4 * t * x_i ** 3 + 2 * t ** 3 - t ** 4

def calc():
    a = 0
    b = 1
    N = 10

    h = dx = (b - a) / N

    x = np.linspace(a, b, N + 1, endpoint=True)
    print(x, dx)

    matrix = np.empty((N + 1, N + 1), dtype=float)
    f = np.empty((N + 1))
    # print(matrix)

    for i in range(1, N):
        for j in range(1, N):
            if 0.5 < abs(i - j) < 1.5:
                matrix[i, j] = - 1. / h + h / 6
            elif i == j:
                matrix[i, j] = 2. / h + 2 * h / 3
            else:
                matrix[i, j] = 0

        # f[i] = h ** 1 / 12 * (x[i] ** 2 + (x[i] + 1) ** 2 + (x[i] + x[i-1] + 1) ** 2 + x[i] ** 2 + (x[i] + 1) ** 2 + (x[i] + x[i+1] + 1) ** 2)
        f[i] = 1. / h / 12 * (integr(x[i], x[i - 1]) + integr(x[i], x[i + 1]))
        # f[i] = h ** 3 / 3 * (2 * x[i] + 1)

    print(matrix[1:N, 1:N])
    print(f[1:N])

    c = np.matmul(np.linalg.inv(matrix[1:N, 1:N]), f[1:N])
    print(c)


# a = np.array([[26 / 9, -7 / 9],
#               [-7 / 9, 26 / 9]])
# f = np.array([-np.sqrt(3) * 6 / 16, -np.sqrt(3) * 6 / 16])
# b = np.matmul(np.linalg.inv(a), f)
# print(b)