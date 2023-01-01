import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import xarray as xr
import numpy as np

runtype = 'TJ'
tend = 'PRECL'
ml = 'rf'
levs_chopped = 0
opts = ['','_5M','_1M','_500k','_50trees']
data_dir = '/glade/scratch/glimon/simplePhysML/19/'+runtype+'/'+tend+'/'+ml+'/'

for opt in opts:
    fn = data_dir+'results'+opt+'.nc'
    d = xr.open_dataset(fn)

    print(opt)

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


    global_R2 = np.average(R2)
    print('weighted R2:',global_R2)

