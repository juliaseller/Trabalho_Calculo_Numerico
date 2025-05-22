import numpy as np
import matplotlib.pyplot as plt
import os

caminho = os.path.join(os.path.dirname(__file__), './Populacao_PresidentePrudente.dat')
dados = np.loadtxt(caminho, skiprows=1) # pular a primeira linha

# print(dados)
x = dados[:, 0] # primeira coluna
y = dados[:, 1] # segunda coluna
plt.scatter(x, y) 
plt.show()


def quadratica(x,y):
    e0 = np.ones(len(x))
    e1 = x
    e2 = x**2
    A = np.array([
        [np.dot(e0, e0), np.dot(e0, e1), np.dot(e0, e2)],
        [np.dot(e1, e0), np.dot(e1, e1), np.dot(e1, e2)],
        [np.dot(e2, e0), np.dot(e2, e1), np.dot(e2, e2)]
    ])
    B = np.array([
        np.dot(e0, y),
        np.dot(e1, y),
        np.dot(e2, y)
    ])

    coef = np.linalg.solve(A, B) # coeficientes da função quadrática, o linag.solve resolve o sistema de equações lineares
    # coef[0] = a0, coef[1] = a1, coef[2] = a2

    y_fit = coef[0] + coef[1]*x + coef[2]*x**2 # função quadrática
    y_fit = np.round(y_fit, 2) # arredondar para duas casas decimais
    plt.plot(x, y_fit, color='blue', label='Ajuste quadrático')
    plt.legend()

    
    print("Valores ajustados:", y_fit)

    
    return y_fit



def cubica(x, y):
    e0 = np.ones(len(x))
    e1 = x
    e2 = x**2
    e3 = x**3
    A = np.array([
        [np.dot(e0, e0), np.dot(e0, e1), np.dot(e0, e2), np.dot(e0, e3)],
        [np.dot(e1, e0), np.dot(e1, e1), np.dot(e1, e2), np.dot(e1, e3)],
        [np.dot(e2, e0), np.dot(e2, e1), np.dot(e2, e2), np.dot(e2, e3)],
        [np.dot(e3, e0), np.dot(e3, e1), np.dot(e3, e2), np.dot(e3, e3)]
    ])
    B = np.array([
        np.dot(e0, y),
        np.dot(e1, y),
        np.dot(e2, y),
        np.dot(e3, y)
    ])

    coef = np.linalg.solve(A, B) # coeficientes da função cúbica
    y_fit = coef[0] + coef[1]*x + coef[2]*x**2 + coef[3]*x**3 # função cúbica
    y_fit = np.round(y_fit, 2) # arredondar para duas casas decimais
    plt.plot(x, y_fit, color='red', label='Ajuste cúbico')
    plt.legend()

    
    print("Valores ajustados:", y_fit)

    
    return y_fit


# aproximar por uma função de 2º grau
# A = np.zeros((3, 3))
# b = np.zeros(3)


# e0 = [1, 1, 1, 1, 1, 1]

# e1 = x

# e2 = np.zeros(6)

# e2 = quadrado(e1, e2, 6)

# C = [e0, e1, e2]

# for k in range(3):
#     B = [C[0][k], C[1][k], C[2][k]]
#     for i in range(3):
#         for j in range(3):
#             A[i][j] = np.dot(B[i], B[j])

# print(A)


# #   for k até 3
# #   B = {C[0][k], C[1][k], C[2][k]}
# #     for i até 3
# #       for j até 3
# #         matriz[i][j] = scalar (B[i], B[j])
