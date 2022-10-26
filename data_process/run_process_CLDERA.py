import numpy as np
import xarray as xr
import functions

inspect_data = True

# Assign runtype
run_type = 'CLDERA' # options: HS, TJ, TJBM
opt = 'passive' # optionsL: allActive
label = 'T_30'
lag = 30 * 4
Ntrain = 100 * 4
Ntest = 10 * 4
xTrainStart = 0
yTrainStart = lag
xTrainStop = Ntrain - lag
yTrainStop = Ntrain
print(xTrainStart, xTrainStop, xTrainStop - xTrainStart)
print(yTrainStart, yTrainStop, yTrainStop - yTrainStart)

xTestStart = -Ntest - lag
yTestStart = -Ntest
xTestStop = -lag
print(xTestStart, xTestStop, xTestStop - xTestStart)
print(yTestStart, - yTestStart)

# Choose attributes
features = ['T','P','U','V','SAI_AOD']

if 'lat' in features:
    LAT = True
else:
    LAT = False

raw = False
chop_levs = 0

out_dir = '/glade/work/glimon/simplePhysML/CLDERA/data/'
data_filename = run_type+'_'+label+'.npz'
inspect_output = '/glade/scratch/glimon/simplePhysML/CLDERA/data/'

print('filename:',data_filename)
print('Features:', features)
print('Label:', label)

# Define feature filenames and assign label filenames
data_files = [
    '/glade/work/glimon/ml_data/'+run_type+'/release_090822/netcdf/'+opt+'/'+'E3SM_ne16_L72_FIDEAL_SAI_'+opt+'.eam.h0.0001-01-01-00000.regrid.2x2.nc',
    '/glade/work/glimon/ml_data/'+run_type+'/release_090822/netcdf/'+opt+'/'+'E3SM_ne16_L72_FIDEAL_SAI_'+opt+'.eam.h0.0001-04-01-00000.regrid.2x2.nc']
#    '/glade/work/glimon/ml_data/'+run_type+'/release_090822/netcdf/'+opt+'/'+'E3SM_ne16_L72_FIDEAL_SAI_'+opt+'.eam.h0.0001-06-30-00000.regrid.2x2.nc']


ds = xr.open_dataset(data_files[0])
D = {}

# Get Weights
lat = ds['lat'].values
if inspect_data:
    lon = ds['lon'].values
    lev = ds['lev'].values
hyai = ds['hyai'].values
hybi = ds['hybi'].values
eta = hyai + hybi
v_weights = eta[1:] - eta[:-1]
l_weights = np.cos(np.deg2rad(lat))


# Load data
dataDict, PS, time = functions.loadData(data_files, features, get_PS_times=True)
print('PS',PS.shape)
print('time',time.shape)

data = functions.arangeArrays(dataDict,features)
print('data',data.shape)
if inspect_data:
    D = functions.fill_dict(D,data,time,lev,lat,lon,features,'raw')

# Roll data axes
data = functions.roll_axes(data)
print(data.shape)
if inspect_data:
    D = functions.fill_dict(D,np.rollaxis(data,3,1),time,lev,lat,lon,features,'1_1')

if not raw:
    # Subtract the mean and divide by the standard deviation
    data, mean, std = functions.scale_ui(data, l_weights=l_weights, v_weights=v_weights, test=True) 
    if inspect_data:
        D = functions.fill_dict(D,np.rollaxis(data,3,1),time,lev,lat,lon,features,'ui')

    # Scale data to range [-1,1] 
    data, dmin, dmax = functions.scale_ab(data,a=0,test=True)
    if inspect_data:
        D = functions.fill_dict(D,np.rollaxis(data,3,1),time,lev,lat,lon,features,'1_1')

if inspect_data:
    xr.Dataset(D).to_netcdf(inspect_output+run_type+'_'+label+'.nc')



trainX = data[xTrainStart:xTrainStop]
trainy = data[yTrainStart:yTrainStop,:,:,:,0:1]
testX = data[xTestStart:xTestStop]
testy = data[yTestStart:,:,:,:,0:1]
PSx = PS[xTestStart:xTestStop]
timex = time[xTestStart:xTestStop]
PSy = PS[yTestStart:]
timey = time[yTestStart:]

# # Reshape data
# trainX = functions.reshape(trainX)
# trainy = functions.reshape(trainy)
# testX = functions.reshape(testX)
# testy, testy_shape = functions.reshape(testy, test=True)
print("split data shapes:")
print(trainX.shape)
print(trainy.shape)
print(testX.shape)
print(testy.shape)

# Save array(s) to disk
np.savez(out_dir+data_filename,
         train_x = trainX,
         train_y = trainy,
         test_x = testX,
         test_y = testy,
         mean = mean,
         std = std,
         dmin = dmin,
         dmax = dmax,
         time = time,
         PS = PS,
         PSx = PSx,
         timex = timex,
         PSy = PSy,
         timey = timey)

# if chop_levs > 0: # Some fields are zero top of the atmosphere, if so we can remove these levels if we want
#     print("chopping")
#     trainX = trainX[:,:,:,chop_levs:,:] 
#     trainy = trainy[:,:,:,chop_levs:,:]
#     testX = testX[:,:,:,chop_levs:,:]
#     testy = testy[:,:,:,chop_levs:,:]
#     print(trainX.shape)
#     print(trainy.shape)
#     print(testX.shape)
#     print(testy.shape)
