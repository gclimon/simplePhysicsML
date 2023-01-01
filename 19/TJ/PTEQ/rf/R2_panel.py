import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np

runtype = 'TJ'
tend = 'PTEQ'
ml = 'rf'
levs_chopped = 6
opt = '_noRH'
data_dir = '/glade/scratch/glimon/simplePhysML/19/'+runtype+'/'+tend+'/'+ml+'/'
fn = data_dir+'results'+opt+'.nc'
d = xr.open_dataset(fn)

print(runtype,tend,ml)

lat = d['lat'].values
lev = d['lev'].values
hyai = d['hyai'].values
hybi = d['hybi'].values
eta = hyai + hybi
v_weights = eta[1:] - eta[:-1]
lat_weights = np.cos(np.deg2rad(lat))
# y = np.average(d[tend].values,axis=3)
# f = np.average(d[tend+'_ML_predicted'].values,axis=3)
y = d[tend].values
f = d[tend+'_ML_predicted'].values
shp = y.shape

y_bar = np.average(np.average(y[:,levs_chopped:,:,:],axis=0),axis=2)
ss_t_array = np.empty((shp[0],shp[1]-levs_chopped,shp[2],shp[3]))
for i in range(shp[0]):
    for j in range(shp[3]):
        ss_t_array[i,:,:,j] = (y[i,levs_chopped:,:,j] - y_bar)

SS_t = np.sum(np.sum(ss_t_array**2,0),2)
SS_r = np.sum(np.sum((y[:,levs_chopped:,:]-f[:,levs_chopped:,:])**2,0),2)
R2 = 1 - SS_r / SS_t

empty = np.zeros((len(lev),len(lat)))
empty[levs_chopped:,:] = R2
R2_da = xr.DataArray(empty, coords=[d['lev'],d['lat']] ,dims=["lev","lat"])
da_dict = {'R2':R2_da,
       'lat': d['lat'],
       'lev': d['lev'],
       'hyai': d['hyai'],
       'hybi': d['hybi'],
       'PS': d['PS']}
DS = xr.Dataset(da_dict)
DS.to_netcdf(data_dir+'R2'+opt+'.nc')

mx = R2.max()
print('max R2:',mx)

global_R2_1 = np.average(R2, axis=0, weights=v_weights[levs_chopped:])
print('max R2_1:',global_R2_1.max())
global_R2 = np.average(global_R2_1, weights=lat_weights)
print('weighted R2:',global_R2)

# fig, ax = plt.subplots(constrained_layout=True)
# CS1 = ax.contourf(lat,lev[levs_chopped:],R2,10)
# # CS1 = ax.contourf(lat,lev[levs_chopped:],R2,[-0.1,0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
# cbar = fig.colorbar(CS1)
# cbar.ax.set_ylabel('R^2')
# ax.set_title('R^2 Dry dT/dt Linear Regression')
# ax.set_xlabel('latitude')
# ax.set_ylabel('Vertical Level')
# ax.invert_yaxis()

# plt.savefig(runtype+'_'+tend+'_'+ml+'.png')
