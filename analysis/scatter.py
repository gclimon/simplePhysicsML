import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
# import matplotlib.transforms as mtransforms
import xarray as xr
import numpy as np

data_dir = '/glade/scratch/glimon/simplePhysML/19/'
fn = 'rf/results.nc'
mdT = xr.open_dataset(data_dir + 'TJ/PTTEND/rf/results_aug.nc')
cdT = xr.open_dataset(data_dir + 'TJBM/PTTEND/rf/results_50trees.nc')
mdQ = xr.open_dataset(data_dir + 'TJ/PTEQ/rf/results_aug.nc')
cdQ = xr.open_dataset(data_dir + 'TJBM/PTEQ/rf/results_50trees.nc')

lat = mdT['lat'].values
lon = mdT['lon'].values
lat_weights = np.cos(np.deg2rad(lat))

mdT_cam = mdT['PTTEND'].values[:,23,:,:] *86400.
mdT_ml = mdT['PTTEND_ML_predicted'].values[:,23,:,:] * 86400.
shp = mdT_cam.shape
print(shp)
mdT_ml  = np.reshape(mdT_ml,(shp[0]*shp[1]*shp[2]))
mdT_cam = np.reshape(mdT_cam,(shp[0]*shp[1]*shp[2]))
mdT_m,mdT_b = np.polyfit(mdT_cam,mdT_ml,1)
mdT_lr = mdT_m * mdT_cam + mdT_b

cdT_cam = cdT['PTTEND'].values[:,23,:,:] *86400.
cdT_ml = cdT['PTTEND_ML_predicted'].values[:,23,:,:] * 86400.
shp = cdT_cam.shape
print(shp)
cdT_ml  = np.reshape(cdT_ml,(shp[0]*shp[1]*shp[2]))
cdT_cam = np.reshape(cdT_cam,(shp[0]*shp[1]*shp[2]))
cdT_m,cdT_b = np.polyfit(cdT_cam,cdT_ml,1)
cdT_lr = cdT_m * cdT_cam + cdT_b

mdQ_cam = mdQ['PTEQ'].values[:,23,:,:] * (1000. * 86400.)
mdQ_ml = mdQ['PTEQ_ML_predicted'].values[:,23,:,:] * (1000. * 86400.)
shp = mdQ_cam.shape
print(shp)
mdQ_ml  = np.reshape(mdQ_ml,(shp[0]*shp[1]*shp[2]))
mdQ_cam = np.reshape(mdQ_cam,(shp[0]*shp[1]*shp[2]))
mdQ_m,mdQ_b = np.polyfit(mdQ_cam,mdQ_ml,1)
mdQ_lr = mdQ_m * mdQ_cam + mdQ_b

cdQ_cam = cdQ['PTEQ'].values[:,23,:,:] * (1000. * 86400.)
cdQ_ml = cdQ['PTEQ_ML_predicted'].values[:,23,:,:] * (1000. * 86400.)
shp = cdQ_cam.shape
print(shp)
cdQ_ml  = np.reshape(cdQ_ml,(shp[0]*shp[1]*shp[2]))
cdQ_cam = np.reshape(cdQ_cam,(shp[0]*shp[1]*shp[2]))
cdQ_m,cdQ_b = np.polyfit(cdQ_cam,cdQ_ml,1)
cdQ_lr = cdQ_m * cdQ_cam + cdQ_b



fig, ax = plt.subplots(2,2)

# trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)

# ax[0,0].set_title('a)', fontfamily='serif', loc='left', fontsize='medium')
ax[0,0].text(-7.0, 61.0, 'a)', 
             # transform=ax[0.0].transAxes + trans,
             fontsize='medium', verticalalignment='top', fontfamily='serif',
             bbox=dict(facecolor='w', edgecolor='none', pad=3.0))
ax[0,0].set_title('Moist dT/dt near 850 hPa')
ax[0,0].scatter(mdT_cam,mdT_ml,s=1,label='ML vs CAM')
ax[0,0].plot(mdT_cam,mdT_cam,'--',color='orange',label='y=x')
ax[0,0].plot(mdT_cam,mdT_lr,'b-',label='Least Squares fit')
ax[0,0].set_xlabel('CAM dT/dt (K/day)')
ax[0,0].set_ylabel('ML dT/dt (K/day)')
ax[0,0].set_xlim(-10,65)
ax[0,0].set_ylim(-10,65)
ax[0,0].legend(fontsize=5,loc='lower right')

# ax[0,1].set_title('b)', fontfamily='serif', loc='left', fontsize='medium')
ax[0,1].text(-13.0, 32.33, 'b)', 
             # transform=ax[0,1].transAxes + trans,
             fontsize='medium', verticalalignment='top', fontfamily='serif',
             bbox=dict(facecolor='w', edgecolor='none', pad=3.0))
ax[0,1].set_title('Conv. dT/dt near 850 hPa')
ax[0,1].scatter(cdT_cam,cdT_ml,s=1,label='ML vs CAM')
ax[0,1].plot(cdT_cam,cdT_cam,'--',color='orange',label='y=x')
ax[0,1].plot(cdT_cam,cdT_lr,'b-',label='Least Squares fit')
ax[0,1].set_xlabel('CAM dT/dt (K/day)')
ax[0,1].set_ylabel('ML dT/dt (K/day)')
ax[0,1].set_xlim(-15,35)
ax[0,1].set_ylim(-15,35)
ax[0,1].legend(fontsize=5,loc='lower right')

ax[1,0].text(-23.0, 22.33, 'c)', 
             # transform=ax[1,0].transAxes + trans,
             fontsize='medium', verticalalignment='top', fontfamily='serif',
             bbox=dict(facecolor='w', edgecolor='none', pad=3.0))
# ax[1,0].set_title('c)', fontfamily='serif', loc='left', fontsize='medium')
ax[1,0].set_title('Moist dq/dt near 850 hPa')
ax[1,0].scatter(mdQ_cam,mdQ_ml,s=1,label='ML vs CAM')
ax[1,0].plot(mdQ_cam,mdQ_cam,'--',color='orange',label='y=x')
ax[1,0].plot(mdQ_cam,mdQ_lr,'b-',label='Least Squares fit')
ax[1,0].set_xlabel('CAM dq/dt (g/kg/day)')
ax[1,0].set_ylabel('ML dq/dt (g/kg/day)')
ax[1,0].set_xlim(-25,25)
ax[1,0].set_ylim(-25,25)
ax[1,0].legend(fontsize=5,loc='lower right')

# ax[1,1].set_title('d)', fontfamily='serif', loc='left', fontsize='medium')
ax[1,1].text(-23.0, 22.33, 'd)', 
             # transform=ax[1,1].transAxes + trans,
             fontsize='medium', verticalalignment='top', fontfamily='serif',
             bbox=dict(facecolor='w', edgecolor='none', pad=3.0))
ax[1,1].set_title('Conv. dq/dt near 850 hPa')
ax[1,1].scatter(cdQ_cam,cdQ_ml,s=1,label='ML vs CAM')
ax[1,1].plot(cdQ_cam,cdQ_cam,'--',color='orange',label='y=x')
ax[1,1].plot(cdQ_cam,cdQ_lr,'b',label='Least Squares fit')
ax[1,1].set_xlabel('CAM dq/dt (g/kg/day)')
ax[1,1].set_ylabel('ML dq/dt (g/kg/day)')
ax[1,1].set_xlim(-25,25)
ax[1,1].set_ylim(-25,25)
ax[1,1].legend(fontsize=5,loc='lower right')

fig.tight_layout()

plt.savefig('scatter.png',dpi=300)
