import numpy as np
import random
import matplotlib.pyplot as plt
import Sens

mv1_bounds = [50, 100]
mv2_bounds = [64, 100]
mv3_bounds = [80, 100]
mv4_bounds = [45, 100]

N = 10**4
M = 12
cv = np.zeros((N, M))

k1 = np.zeros(N)
k2 = np.zeros(N)

nn = 0
for i in range(1, 11):
    mv1 = mv1_bounds[0] + (i/10)*(mv1_bounds[1]-mv1_bounds[0])
    for j in range(1, 11):
        mv2 = mv2_bounds[0] + (j/10) * (mv2_bounds[1] - mv2_bounds[0])
        for k in range(1, 11):
            mv3 = mv3_bounds[0] + (k/10) * (mv3_bounds[1] - mv3_bounds[0])
            for w in range(1, 11):
                mv4 = mv4_bounds[0] + (w/10) * (mv4_bounds[1] - mv4_bounds[0])

                cv[nn, 0] = 1.9 * mv4
                cv[nn, 1] = 12 * mv4
                cv[nn, 2] = 2.5 * mv3
                cv[nn, 3] = 0.31 * mv3 - 0.02 * mv4
                cv[nn, 4] = 0.02 * mv3 + 0.01 * mv4
                cv[nn, 5] = mv1 + mv4
                cv[nn, 6] = -0.004 * mv1 + 0.034 * mv4
                cv[nn, 7] = 0.1 * mv2 + 0.2 * mv3
                cv[nn, 8] = 1.25 * mv2
                cv[nn, 9] = 1.2 * mv1

                cv[nn, 10] = 0.26 * cv[nn, 0] - 0.0004 * cv[nn, 2]**2 + 6.17*10**(-5)*cv[nn, 3]**4 + 0.9 * cv[nn, 4]**3
                cv[nn, 11] = 1.6*10**(-5) * cv[nn, 1]**2 + 0.5 * cv[nn, 6]**4 + 0.8 * cv[nn, 7] - 1.44 * 10**(-5) * cv[nn, 9]**3
                nn += 1

cv = cv[cv[:, 10].argsort()]

t = np.zeros(N)

for i in range(N):
    t[i] = i*60

np.savetxt('lab_cv.csv', np.vstack([t, cv[:, :-2].T]).T, delimiter=',')
np.savetxt('lab_k1.csv', np.vstack([t, cv[:, 10]]).T, delimiter=',')
np.savetxt('lab_k2.csv', np.vstack([t, cv[:, 11]]).T, delimiter=',')
print(k1)


