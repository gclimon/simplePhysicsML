import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import xarray as xr
import numpy as np


data_dir = '/glade/work/glimon/simplePhysML/19/'
fn = 'rf/results.nc'
mdT = xr.open_dataset(data_dir + 'TJ/PTTEND/' + fn)
cdT = xr.open_dataset(data_dir + 'TJBM/PTTEND/' + fn)
mdQ = xr.open_dataset(data_dir + 'TJ/PTEQ/' + fn)
cdQ = xr.open_dataset(data_dir + 'TJBM/PTEQ/' + fn)

lat = mdT['lat'].values
lon = mdT['lon'].values
lat_weights = np.cos(np.deg2rad(lat))


mdT_diff = mdT['PTTEND_difference'].values[:,23,:,:] *86400.
shp = mdT_diff.shape
print(shp)
mdT_diff  = np.reshape(mdT_diff,(shp[0]*shp[1]*shp[2]))

cdT_diff = cdT['PTTEND_difference'].values[:,23,:,:] *86400.
shp = cdT_diff.shape
print(shp)
cdT_diff  = np.reshape(cdT_diff,(shp[0]*shp[1]*shp[2]))

mdQ_diff = mdQ['PTEQ_difference'].values[:,23,:,:] * (1000. * 86400.)
shp = mdQ_diff.shape
print(shp)
mdQ_diff  = np.reshape(mdQ_diff,(shp[0]*shp[1]*shp[2]))

cdQ_diff = cdQ['PTEQ_difference'].values[:,23,:,:] * (1000. * 86400.)
shp = cdQ_diff.shape
print(shp)
cdQ_diff  = np.reshape(cdQ_diff,(shp[0]*shp[1]*shp[2]))

n_bins = 100
# lgbins0 = np.logspace(start=np.log(mdT_diff.min()), stop=np.log(mdT_diff.min()), num=500)
# lgbins1 = np.logspace(start=np.log(cdT_diff.min()), stop=np.log(cdT_diff.min()), num=500)
# lgbins2 = np.logspace(start=np.log(mdQ_diff.min()), stop=np.log(mdQ_diff.min()), num=500)
# lgbins3 = np.logspace(start=np.log(cdQ_diff.min()), stop=np.log(cdQ_diff.min()), num=500)

fig, ax = plt.subplots(2,2)
ax[0,0].text(-15.0, 1e6, 'a)', 
             fontsize='medium', verticalalignment='top', fontfamily='serif',
             bbox=dict(facecolor='w', edgecolor='none', pad=3.0))
h1 = ax[0,0].hist(mdT_diff, bins=n_bins)
h1_percent = np.asarray(h1[0])/len(mdT_diff)
sorted_h1_ind = np.argsort(h1_percent)
percent_h1 = 0.
i_h1 = []
for i in sorted_h1_ind[::-1]:
    if percent_h1 > 0.95:
        break
    else:
        percent_h1 += h1_percent[i]
        i_h1.append(i)
print(percent_h1)
print(i_h1)
h1_min = min(i_h1)
h1_max = max(i_h1)
ax[0,0].axvline(x=h1[1][h1_min], color='k', linestyle='dashed', linewidth=0.5,label='>97%')
ax[0,0].axvline(x=h1[1][h1_max], color='k', linestyle='dashed', linewidth=0.5)
# ax[0,0].set_xticks(list(ax[0,0].get_xticks()) + [-15., -5., 5., 15.])
ax[0,0].xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax[0,0].set_title('Moist dT/dt near 850 hPa')
ax[0,0].set_ylabel('N')
ax[0,0].set_xlabel('ML - CAM (K/day)')
ax[0,0].set_yscale("log")
ax[0,0].legend(fontsize=6)
# ax[0,0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(mdT_diff)))

ax[0,1].text(-15.0, 1e6, 'b)', 
             fontsize='medium', verticalalignment='top', fontfamily='serif',
             bbox=dict(facecolor='w', edgecolor='none', pad=3.0))
h2 = ax[0,1].hist(cdT_diff, bins=n_bins)
h2_percent = np.asarray(h2[0])/len(mdT_diff)
sorted_h2_ind = np.argsort(h2_percent)
percent_h2 = 0.
i_h2 = []
for i in sorted_h2_ind[::-1]:
    if percent_h2 > 0.95:
        break
    else:
        percent_h2 += h2_percent[i]
        i_h2.append(i)
print(percent_h2)
print(i_h2)
h2_min = min(i_h2)
h2_max = max(i_h2)
ax[0,1].axvline(x=h2[1][h2_min], color='k', linestyle='dashed', linewidth=0.5, label='>95%')
ax[0,1].axvline(x=h2[1][h2_max], color='k', linestyle='dashed',linewidth=0.5)
# ax[0,1].set_xticks(list(ax[0,1].get_xticks()) + [-15., -5., 5., 15.])
# ax[0,1].xaxis.set_major_locator(plt.MaxNLocator(3))
ax[0,1].xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax[0,1].set_title('Convection dT/dt near 850 hPa')
ax[0,1].set_ylabel('N')
ax[0,1].set_xlabel('ML - CAM (K/day)')
ax[0,1].set_yscale("log")
ax[0,1].legend(fontsize=6)
# ax[0,1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(cdT_diff)))

ax[1,0].text(-12.0, 1e6, 'c)', 
             fontsize='medium', verticalalignment='top', fontfamily='serif',
             bbox=dict(facecolor='w', edgecolor='none', pad=3.0))
h3 = ax[1,0].hist(mdQ_diff, bins=n_bins)
h3_percent = np.asarray(h3[0])/len(mdT_diff)
sorted_h3_ind = np.argsort(h3_percent)
percent_h3 = 0.
i_h3 = []
for i in sorted_h3_ind[::-1]:
    if percent_h3 > 0.95:
        break
    else:
        percent_h3 += h3_percent[i]
        i_h3.append(i)
print(percent_h3)
print(i_h3)
h3_min = min(i_h3)
h3_max = max(i_h3)
ax[1,0].axvline(x=h3[1][h3_min], color='k', linestyle='dashed', linewidth=0.5,label='>95%')
ax[1,0].axvline(x=h3[1][h3_max], color='k', linestyle='dashed', linewidth=0.5)
ax[1,0].set_title('Moist dq/dt near 850 hPa')
ax[1,0].set_ylabel('N')
ax[1,0].set_xlabel('ML - CAM (g/kg/day)')
ax[1,0].set_yscale("log")
ax[1,0].legend(fontsize=6)
# ax[1,0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(mdQ_diff)))

ax[1,1].text(-15.0, 1e6, 'd)', 
             fontsize='medium', verticalalignment='top', fontfamily='serif',
             bbox=dict(facecolor='w', edgecolor='none', pad=3.0))
h4 = ax[1,1].hist(cdQ_diff, bins=n_bins)
h4_percent = np.asarray(h4[0])/len(mdT_diff)
sorted_h4_ind = np.argsort(h4_percent)
percent_h4 = 0.
i_h4 = []
for i in sorted_h4_ind[::-1]:
    if percent_h4 > 0.95:
        break
    else:
        percent_h4 += h4_percent[i]
        i_h4.append(i)
print(percent_h4)
print(i_h4)
h4_min = min(i_h4)
h4_max = max(i_h4)
ax[1,1].axvline(x=h4[1][h4_min], color='k', linestyle='dashed', linewidth=0.5,label='>95%')
ax[1,1].axvline(x=h4[1][h4_max], color='k', linestyle='dashed', linewidth=0.5)
# ax[1,1].set_xticks(list(ax[1,1].get_xticks()) + [-15., -5., 5.])
# ax[1,1].xaxis.set_major_locator(plt.MaxNLocator(5))
ax[1,1].xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax[1,1].set_title('Convection dq/dt near 850 hPa')
ax[1,1].set_ylabel('N')
ax[1,1].set_xlabel('ML - CAM (g/kg/day)')
ax[1,1].set_yscale("log")
ax[1,1].legend(fontsize=6)
# ax[1,1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(cdQ_diff)))

fig.tight_layout()
plt.savefig('hist.png',dpi=300)
