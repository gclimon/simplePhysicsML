import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np

runtype = 'TJ'
tend = 'PTTEND'
ml = 'rf'
levs_chopped = 12
opts = ['_aug','_noRH','_5M','_1M','_500k']
data_dir = '/glade/scratch/glimon/simplePhysML/19/'+runtype+'/'+tend+'/'+ml+'/'
print(runtype,tend,ml)

for opt in opts:
    fn =data_dir+'results'+opt+'.nc'
    d = xr.open_dataset(fn)

    print(opt)

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

    global_R2_1 = np.average(R2, axis=0, weights=v_weights[levs_chopped:])

    global_R2 = np.average(global_R2_1, weights=lat_weights)
    print('weighted R2:',global_R2)

