import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np

R2_mdT = np.array([0.8805, 0.8690, 0.8433, 0.8288])
N_mdT = np.array([15000000, 5000000, 1000000, 500000 ]) *  1e-6
R2_nRH_mdT = np.array([0.7499])
N_nRH_mdT = np.array([15000000]) *  1e-6

R2_mdq = np.array([0.8797, 0.8579, 0.8170, 0.7731])
N_mdq = np.array([20000000, 5000000, 1000000, 500000 ]) *  1e-6
R2_nRH_mdq = np.array([0.7431])
N_nRH_mdq = np.array([20000000]) *  1e-6

R2_cdT = np.array([0.8295, 0.8128, 0.7711, 0.7389])
N_cdT = np.array([15000000, 5000000, 1000000, 500000 ]) *  1e-6
R2_nRH_cdT = np.array([0.7308])
N_nRH_cdT = np.array([15000000]) *  1e-6

R2_cdq = np.array([0.858,   0.827,   0.772,   0.737])
N_cdq = np.array([20000000, 5000000, 1000000, 500000 ]) *  1e-6
R2_nRH_cdq = np.array([0.735])
N_nRH_cdq = np.array([20000000]) *  1e-6

fig, ax = plt.subplots(1,1)
ax.set_title('R2 Variation')
ax.set_xlabel('N samples (in millions)')
ax.set_ylabel('R2')

ax.scatter(N_mdT,R2_mdT,color='k')
ax.scatter(N_nRH_mdT,R2_nRH_mdT,color='k',marker='x',label='No RH')
ax.plot(N_mdT,R2_mdT,color='k',label='Moist dT/dt')

ax.scatter(N_cdT,R2_cdT,color='b')
ax.scatter(N_nRH_cdT,R2_nRH_cdT,color='b',marker='x')
ax.plot(N_cdT,R2_cdT,color='b',label='Convection dT/dt')

ax.scatter(N_mdq,R2_mdq,color='r')
ax.scatter(N_nRH_mdq,R2_nRH_mdq,color='r',marker='x')
ax.plot(N_mdq,R2_mdq,color='r',label='Moist dq/dt')

ax.scatter(N_cdq,R2_cdq,color='y')
ax.scatter(N_nRH_cdq,R2_nRH_cdq,color='y',marker='x')
ax.plot(N_cdq,R2_cdq,color='y',label='Convection dq/dt')

#ax.invert_xaxis()
fig.tight_layout()
ax.legend()

plt.savefig('trainTest.png',dpi=500)
