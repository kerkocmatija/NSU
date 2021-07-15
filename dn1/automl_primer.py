import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def generiraj(n, povprecje, std):
    a = np.random.randn(n) * std + povprecje
    a = a[a <= 1.0]
    a = a[a >= 0.0]
    return a


alg1 = generiraj(1000, 0.8, 0.2)
alg2 = generiraj(600, 0.5, 0.3)
alg3 = generiraj(500, 0.3, 0.4)

_, axes = plt.subplots(nrows=3, ncols=1)

plt.xlim([0, 1])
sns.distplot(alg1, label="alg1", rug=True, bins=10, ax=axes[0])
axes[0].legend()
sns.distplot(alg2, label="alg2", rug=True, bins=10, ax=axes[1])
axes[1].legend()
sns.distplot(alg3, label="alg3", rug=True, bins=10, ax=axes[2])
axes[2].legend()

print(alg1[0])

plt.show()
