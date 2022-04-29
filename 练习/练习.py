import numpy as np

import random
import numpy as np

x = []
y = np.linspace(800, 1200, 9)
for i in range(9):
    a = np.random.normal(loc=0.0, scale=0.0015, size=None)
    a = 1 + a
    x.append(a)
z = x * y
print(z)
