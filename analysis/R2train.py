import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np

R2 = np.array([0.894, 0.891, 0.888, 0.876, 0.842, 0.825, 0.733, 0.478 ])
N = np.array([20000000, 15000000, 10000000, 5000000, 1000000, 500000, 50000, 5000 ])

fig, ax = plt.subplots(1,1)
ax.set_title('Moist dq/dt')
ax.scatter(np.log10(N),R2)
ax.plot(np.log10(N),R2,color='k')
ax.set_xlabel('log(N)')
ax.set_ylabel('R2')
ax.invert_xaxis()
fig.tight_layout()

plt.savefig('trainTest.png',dpi=300)
