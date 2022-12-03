import numpy as np
import random
import matplotlib.pyplot as plt
import Sens

mv1_bounds = [50, 100]
mv2_bounds = [64, 100]
mv3_bounds = [80, 100]
mv4_bounds = [45, 100]

N = 1000
M = 10
cv = np.zeros((N, M))
cv_v = np.zeros((N, 4))
k1 = np.zeros(N)
k2 = np.zeros(N)

for i in range(1, 21):
    mv1 = mv1_bounds[0] + (i/20)*(mv1_bounds[1]-mv1_bounds[0])
    for j in range(1, 21):
        mv2 = mv2_bounds[0] + (j/20) * (mv2_bounds[1] - mv2_bounds[0])
        for k in range(1, 21):
            mv3 = mv3_bounds[0] + (k/20) * (mv3_bounds[1] - mv3_bounds[0])
            for w in range(1, 21):
                mv4 = mv4_bounds[0] + (w/20) * (mv4_bounds[1] - mv4_bounds[0])

                cv[i, 0] = 1.9 * mv4
                cv[i, 1] = 12 * mv4
                cv[i, 2] = 2.5 * mv3
                cv[i, 3] = 0.32 * mv3 - 0.02 * mv4
                cv[i, 4] = 0.02 * mv3 + 0.01 * mv4
                cv[i, 5] = mv1 + mv4
                cv[i, 6] = -0.004 * mv1 + 0.035 * mv4
                cv[i, 7] = 0.1 * mv2 + 0.2 * mv3
                cv[i, 8] = 1.25 * mv2
                cv[i, 9] = 1.2 * mv1

    k1[i] = 0.26 * cv[i, 0] - 0.0004 * cv[i, 2]**2 + 6.17*10**(-5)*cv[i, 3]**4 + 0.9 * cv[i, 4]**3
    k2[i] = 1.7*10**(-5) * cv[i, 1]**2 + 0.5 * cv[i, 6]**4 + 0.8 * cv[i, 7] - 1.44 * 10**(-5) * cv[i, 9]**3

print(max(k2))
