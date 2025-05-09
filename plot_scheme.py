import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1])
y = np.array([0, 0])

plt.figure(figsize=(10, 2))

plt.xticks([0, 1])
plt.yticks([])

plt.xlabel("x")

plt.ylim(-0.05, 0.1)

plt.plot(0, 0, marker='o', markersize=20, color='black', fillstyle='none')
plt.plot(1, 0, marker='o', markersize=20, color='black', fillstyle='full')

# plt.box(False)

plt.show()