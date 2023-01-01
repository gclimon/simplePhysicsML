import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np

data_dir = '/glade/scratch/glimon/simplePhysML/19/'
fn = 'rf/results.nc'
mPL = xr.open_dataset(data_dir + 'TJ/PRECL/rf/results_aug.nc')
cPL = xr.open_dataset(data_dir + 'TJBM/PRECL/rf/results_50trees.nc')
cPC = xr.open_dataset(data_dir + 'TJBM/PRECC/rf/results_50trees.nc')

lat = mPL['lat'].values
lon = mPL['lon'].values
lat_weights = np.cos(np.deg2rad(lat))

mPL_cam = mPL['PRECL'].values * (1000.*86400.)
mPL_ml = mPL['PRECL_ML_predicted'].values * (1000.*86400.)
shp = mPL_cam.shape
print(shp)
mPL_ml  = np.reshape(mPL_ml,(shp[0]*shp[1]*shp[2]))
mPL_cam = np.reshape(mPL_cam,(shp[0]*shp[1]*shp[2]))
mPL_m,mPL_b = np.polyfit(mPL_cam,mPL_ml,1)
mPL_lr = mPL_m * mPL_cam + mPL_b

cPL_cam = cPL['PRECL'].values * (1000.*86400.)
cPL_ml = cPL['PRECL_ML_predicted'].values * (1000.*86400.)
shp = cPL_cam.shape
print(shp)
cPL_ml  = np.reshape(cPL_ml,(shp[0]*shp[1]*shp[2]))
cPL_cam = np.reshape(cPL_cam,(shp[0]*shp[1]*shp[2]))
cPL_m,cPL_b = np.polyfit(cPL_cam,cPL_ml,1)
cPL_lr = cPL_m * cPL_cam + cPL_b

cPC_cam = cPC['PRECC'].values * (1000.*86400.)
cPC_ml = cPC['PRECC_ML_predicted'].values * (1000.*86400.)
shp = cPC_cam.shape
print(shp)
cPC_ml  = np.reshape(cPC_ml,(shp[0]*shp[1]*shp[2]))
cPC_cam = np.reshape(cPC_cam,(shp[0]*shp[1]*shp[2]))
cPC_m,cPC_b = np.polyfit(cPC_cam,cPC_ml,1)
cPC_lr = cPC_m * cPC_cam + cPC_b

fig, ax = plt.subplots(2,2)
ax[0,0].text(20.0, 425.0, 'a)', 
             # transform=ax[0.0].transAxes + trans,
             fontsize='medium', verticalalignment='top', fontfamily='serif',
             bbox=dict(facecolor='w', edgecolor='none', pad=3.0))
ax[0,0].set_title('Moist Large Scale Precip')
ax[0,0].scatter(mPL_cam,mPL_ml,s=1,label='ML vs CAM')
ax[0,0].plot(mPL_cam,mPL_cam,'--',color='orange',label='y=x')
ax[0,0].plot(mPL_cam,mPL_lr,'b-',label='Least Squares fit')
ax[0,0].set_xlabel('CAM PRECL (mm/day)')
ax[0,0].set_ylabel('ML PRECL (mm/day)')
ax[0,0].set_xlim(-5,450)
ax[0,0].set_ylim(-5,450)
ax[0,0].legend(fontsize=5, loc='lower right')

ax[0,1].text(3.516, 141.484, 'b)', 
             # transform=ax[0.0].transAxes + trans,
             fontsize='medium', verticalalignment='top', fontfamily='serif',
             bbox=dict(facecolor='w', edgecolor='none', pad=3.0))
ax[0,1].set_title('Convection Large Scale Precip')
ax[0,1].scatter(cPL_cam,cPL_ml,s=1,label='ML vs CAM')
ax[0,1].plot(cPL_cam,cPL_cam,'--',color='orange',label='y=x')
ax[0,1].plot(cPL_cam,cPL_lr,'b-',label='Least Squares fit')
ax[0,1].set_xlabel('CAM PRECL (mm/day)')
ax[0,1].set_ylabel('ML PRECL (mm/day)')
ax[0,1].set_xlim(-5,150)
ax[0,1].set_ylim(-5,150)
ax[0,1].legend(fontsize=5, loc='lower right')

ax[1,0].axis('off')

ax[1,1].text(-2.47, 37.53, 'c)', 
             # transform=ax[0.0].transAxes + trans,
             fontsize='medium', verticalalignment='top', fontfamily='serif',
             bbox=dict(facecolor='w', edgecolor='none', pad=3.0))
ax[1,1].set_title('Convection Convective Precip')
ax[1,1].scatter(cPC_cam,cPC_ml,s=1,label='ML vs CAM')
ax[1,1].plot(cPC_cam,cPC_cam,'--',color='orange',label='y=x')
ax[1,1].plot(cPC_cam,cPC_lr,'b-',label='Least Squares fit')
ax[1,1].set_xlabel('CAM PRECC (mm/day)')
ax[1,1].set_ylabel('ML PRECC (mm/day)')
ax[1,1].set_xlim(-5,40)
ax[1,1].set_ylim(-5,40)
ax[1,1].legend(fontsize=5, loc='lower right')

fig.tight_layout()

plt.savefig('scatter_precip.png',dpi=300)
