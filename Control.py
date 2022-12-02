import numpy as np
import matplotlib.pyplot as plt
import time

# Исходные данные
L = 300                 # м
A = 62.5                # м^2
kappa = 0.7             # м^2/c
po = 2*10**7            # Па
k = 10**(-13)           # м^2
mu = 10**(-3)           # Па*с
B = 1.1                 # м^3/м^3
Q = 2/(24*60*60)       # м^3/c
T = 100*24*60*60        # с

# Параметры прямой задачи
# (I) изменяемые
M = 20
N = 1000
# (II) зависимые от (I)
h = L/M
tau = T/N

a = - kappa*tau/h**2
b = 1 - 2*a
d = 2*mu*Q*B*h/(A*k)

# Массив для хранения значений на каждом слою
p = np.zeros((N+1, M+1))

# Начальные условия
for i in range(M+1):
    p[0, i] = po
# Граничные условия
for j in range(1, N+1):
    p[j, M] = po

# Предупреждение
if a == 0:
    print("a=0")

# Формирование матрицы A
A = np.zeros((M, M))

# Первая строчка
A[0][0] = -2
A[0][1] = 4 + b/a

for i in range(1, M):
    A[i][i-1] = a
    A[i][i] = b
    if i < M-1:
        A[i][i+1] = a

# Формирование правой части
for j in range(0, N):
    f = np.zeros((M))
    f[0] = d + p[j][1]/a
    for i in range(1, M-1):
        f[i] = p[j][i]
    f[M-1] = p[j][M-1] - a*p[j+1][M]

    alpha = np.zeros((M))
    beta = np.zeros((M))
    alpha[1] = - A[0][1]/A[0][0]
    beta[1] = f[0]/A[0][0]
    for i in range(1, M-1):
        alpha[i+1] = - A[i][i+1]/(A[i][i-1]*alpha[i]+A[i][i])
        beta[i+1] = (f[i]-A[i][i-1]*beta[i])/(A[i][i-1]*alpha[i]+A[i][i])
    p[j+1][M-1] = (f[M-1]-A[M-1][M-2]*beta[M-1])/(A[M-1][M-1]+A[M-1][M-2]*alpha[M-1])
    for i in range(M-2, -1, -1):
        p[j+1][i] = alpha[i+1]*p[j+1][i+1]+beta[i+1]

x = np.linspace(0., L, M+1)

for j in range(N+1):
    plt.plot(x, p[j][:])
    st = "Значение давления на " + str(int(j * tau/(60*60*24))) + "-ые сутки"
    plt.title(st)
plt.show()
