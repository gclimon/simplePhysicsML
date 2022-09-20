import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import xarray as xr
import numpy as np

data_dir = '/glade/work/glimon/simplePhysML/19/'
fn = 'rf/results.nc'
mPL = xr.open_dataset(data_dir + 'TJ/PRECL/' + fn)
cPL = xr.open_dataset(data_dir + 'TJBM/PRECL/' + fn)
cPC = xr.open_dataset(data_dir + 'TJBM/PRECC/' + fn)

lat = mPL['lat'].values
lon = mPL['lon'].values
lat_weights = np.cos(np.deg2rad(lat))


mPL_diff = mPL['PRECL_difference'].values * (1000.*86400.)
shp = mPL_diff.shape
print(shp)
mPL_diff  = np.reshape(mPL_diff,(shp[0]*shp[1]*shp[2]))

cPL_diff = cPL['PRECL_difference'].values * (1000.*86400.)
shp = cPL_diff.shape
print(shp)
cPL_diff  = np.reshape(cPL_diff,(shp[0]*shp[1]*shp[2]))

cPC_diff = cPC['PRECC_difference'].values * (1000.*86400.)
shp = cPC_diff.shape
print(shp)
cPC_diff  = np.reshape(cPC_diff,(shp[0]*shp[1]*shp[2]))

n_bins = 100

fig, ax = plt.subplots(2,2)
ax[0,0].text(-45.0, 1e6, 'a)', 
             fontsize='medium', verticalalignment='top', fontfamily='serif',
             bbox=dict(facecolor='w', edgecolor='none', pad=3.0))
h1 = ax[0,0].hist(mPL_diff, bins=n_bins)
h1_percent = np.asarray(h1[0])/len(mPL_diff)
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
h1_max = max(i_h1) + 1
ax[0,0].axvline(x=h1[1][h1_min], color='k', linestyle='dashed', linewidth=0.5, label='>97%')
ax[0,0].axvline(x=h1[1][h1_max], color='k', linestyle='dashed', linewidth=0.5)
# ax[0,0].set_xticks(list(ax[0,0].get_xticks()) + [-20., -15., -10, -5., 5., 10., 15., 20.])
# ax[0,0].xaxis.set_major_locator(plt.MaxNLocator(5))
ax[0,0].xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax[0,0].set_title('Moist Large Scale Precip')
ax[0,0].set_ylabel('N')
ax[0,0].set_xlabel('ML - CAM (mm/day)')
ax[0,0].set_yscale('log')
ax[0,0].legend(fontsize=6)

ax[0,1].text(-45.0, 1e6, 'b)', 
             fontsize='medium', verticalalignment='top', fontfamily='serif',
             bbox=dict(facecolor='w', edgecolor='none', pad=3.0))
h2 = ax[0,1].hist(mPL_diff, bins=n_bins)
h2_percent = np.asarray(h2[0])/len(mPL_diff)
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
h2_max = max(i_h2) + 1
ax[0,1].axvline(x=h2[1][h2_min], color='k', linestyle='dashed', linewidth=0.5, label='>97%')
ax[0,1].axvline(x=h2[1][h2_max], color='k', linestyle='dashed', linewidth=0.5)
# ax[0,1].set_xticks(list(ax[0,1].get_xticks()) + [-20., -15., -10, -5., 5., 10., 15., 20.])
# ax[0,1].xaxis.set_major_locator(plt.MaxNLocator(5))
ax[0,1].xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax[0,1].set_title('Convection Large Scale Precip')
ax[0,1].set_ylabel('N')
ax[0,1].set_xlabel('ML - CAM (mm/day)')
ax[0,1].set_yscale('log')
ax[0,1].legend(fontsize=6)

ax[1,0].axis('off')

ax[1,1].text(-13.0, 1e6, 'c)', 
             fontsize='medium', verticalalignment='top', fontfamily='serif',
             bbox=dict(facecolor='w', edgecolor='none', pad=3.0))
h3 = ax[1,1].hist(cPC_diff, bins=n_bins)
h3_percent = np.asarray(h3[0])/len(mPL_diff)
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
ax[1,1].axvline(x=h3[1][h3_min], color='k', linestyle='dashed', linewidth=0.5, label='>95%')
ax[1,1].axvline(x=h3[1][h3_max], color='k', linestyle='dashed', linewidth=0.5)
ax[1,1].xaxis.set_minor_locator(ticker.MultipleLocator(5))
# ax[1,1].set_xticks(list(ax[1,1].get_xticks()) + [-5., 5.])
# ax[1,1].xaxis.set_major_locator(plt.MaxNLocator(5))
ax[1,1].set_title('Convection Convective Precip')
ax[1,1].set_ylabel('N')
ax[1,1].set_xlabel('ML - CAM (mm/day)')
ax[1,1].set_yscale('log')
ax[1,1].legend(fontsize=6)

fig.tight_layout()

plt.savefig('precip_hist.png',dpi=300)
