import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np

runtype = 'TJBM'
tend = 'PRECC'
ml = 'rf'
levs_chopped = 0
opt = '_50trees'
data_dir = '/glade/scratch/glimon/simplePhysML/19/'+runtype+'/'+tend+'/'+ml+'/'
fn = data_dir+'results'+opt+'.nc'
d = xr.open_dataset(fn)

print(runtype,tend,ml)

lat = d['lat'].values
lon = d['lon'].values
lat_weights = np.cos(np.deg2rad(lat))
# y = np.average(d[tend].values,axis=3)
# f = np.average(d[tend+'_ML_predicted'].values,axis=3)
y = d[tend].values
f = d[tend+'_ML_predicted'].values
shp = y.shape

y_bar = np.average(np.average(y,axis=0),axis=1)
ss_t_array = np.empty(shp)
for i in range(shp[0]):
    for j in range(shp[2]):
        ss_t_array[i,:,j] = (y[i,:,j] - y_bar)

SS_t = np.sum(np.sum(ss_t_array**2,0),1)
SS_r = np.sum(np.sum((y-f)**2,0),1)
R2 = 1 - SS_r / SS_t

empty = np.zeros((len(lat)))
empty[:] = R2
R2_da = xr.DataArray(empty, coords=[d['lat']] ,dims=["lat"])
da_dict = {'R2':R2_da,
       'lat': d['lat'],
       'lon': d['lon'],
       'hyai': d['hyai'],
       'hybi': d['hybi'],
       'PS': d['PS']}
DS = xr.Dataset(da_dict)
DS.to_netcdf(data_dir+'R2'+opt+'.nc')

mx = R2.max()
print('max R2:',mx)

global_R2 = np.average(R2)
print('weighted R2:',global_R2)

fig, ax = plt.subplots(constrained_layout=True)
P1 = ax.plot(lat,R2)
ax.set_title('Random Forest $R^2$ (convection PRECC)')
ax.set_xlabel('Latitude (deg)')
ax.set_ylabel('$R^2$')

ax.text(0.05, 0.95, "g)", transform=ax.transAxes, fontsize=24,
        verticalalignment='top')

plt.savefig(runtype+'_'+tend+'R2'+opt+'.png')
