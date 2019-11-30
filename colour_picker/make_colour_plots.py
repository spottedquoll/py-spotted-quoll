import matplotlib.pyplot as plt
import numpy as np

height = [5, 5, 5, 5, 5]
bars = ('A', 'B', 'C', 'D', 'E')
y_pos = np.arange(len(bars))

plt.bar(y_pos, height, color=('cloudy blue', 0.4, 0.6, 0.6))
plt.xticks(y_pos, bars)
plt.show()