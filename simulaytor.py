import numpy as np
import random
import Sens

mv1_bounds = [50, 100]
mv2_bounds = [64, 100]
mv3_bounds = [80, 100]
mv4_bounds = [45, 100]

N = 1000
M = 10
cv = np.zeros((N, M))
k = np.zeros(N)

for i in range(N):
    mv1 = mv1_bounds[0] + random.random()*(mv1_bounds[1]-mv1_bounds[0])
    mv2 = mv2_bounds[0] + random.random() * (mv2_bounds[1] - mv2_bounds[0])
    mv3 = mv3_bounds[0] + random.random() * (mv3_bounds[1] - mv3_bounds[0])
    mv4 = mv4_bounds[0] + random.random() * (mv4_bounds[1] - mv4_bounds[0])

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

    k[i] = 0.26 * cv[i, 0] - 0.1 * cv[i, 2] + 0.83 * cv[i, 3] + 8.3 * cv[i, 4]

name_finish_x_short, name_finish_x, yy, b, mass_on = Sens.proposed_model(cv, k, ['cv1', 'cv2', 'cv3', 'cv4', 'cv5', 'cv6', 'cv7', 'cv8', 'cv9', 'cv10'])

print(name_finish_x)