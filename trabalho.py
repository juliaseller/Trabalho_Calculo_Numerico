import numpy as np
import matplotlib.pyplot as plt
import os

caminho = os.path.join(os.path.dirname(__file__), 'Corrente_Tensao.dat')
dados = np.loadtxt(caminho)

# print(dados)
x = dados[:, 0]
y = dados[:, 1]
plt.scatter(x, y)
plt.show()


def prodEscalar(a, b, n):
    soma = 0
    i = 0
    for i in range(n):
        soma += a[i]*b[i]
    return soma


def quadrado(e1, e2, n):
    for i in range(n):
        e2[i] = e1[i]**2
    return e2


# aproximar por uma função de 2º grau
A = np.zeros((3, 3))
b = np.zeros(3)


e0 = [1, 1, 1, 1, 1, 1]

e1 = x

e2 = np.zeros(6)

e2 = quadrado(e1, e2, 6)

C = [e0, e1, e2]

for k in range(3):
    B = [C[0][k], C[1][k], C[2][k]]
    for i in range(3):
        for j in range(3):
            A[i][j] = prodEscalar(B[i], B[j], 6)

print(A)


#   for k até 3
#   B = {C[0][k], C[1][k], C[2][k]}
#     for i até 3
#       for j até 3
#         matriz[i][j] = scalar (B[i], B[j])
