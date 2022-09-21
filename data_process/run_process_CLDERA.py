import numpy as np
import xarray as xr
import functions

inspect_data = False

# Assign runtype
run_type = 'CLDERA' # options: HS, TJ, TJBM
label = ''

# Choose attributes
features = ['T','P','lat']

if 'lat' in features:
    LAT = True
else:
    LAT = False

n_train = 
raw = True
chop_levs = 0

out_dir = '/glade/work/glimon/simplePhysML//data/'
data_filename = (run_type+'_'+label+'.npz')

print('filename:',data_filename)
print('Resolution:', res)
print('Features:', features)
print('Label:', label)

# Define feature filenames and assign label filenames
train_feature_files = [
    '/glade/work/glimon/ml_data/'+run_type+'/'+'',
    '/glade/work/glimon/ml_data/'+run_type+'/'+'',
    '/glade/work/glimon/ml_data/'+run_type+'/'+'',
    '/glade/work/glimon/ml_data/'+run_type+'/'+'',
    '/glade/work/glimon/ml_data/'+run_type+'/'+'']

test_feature_files = [    
    '/glade/work/glimon/ml_data/'+run_type+'/'+'']


train_label_files = []
for f in train_feature_files:
    train_label_files.append(f.replace('.h0.','.h1.'))
test_label_files = []
for f in test_feature_files:
    test_label_files.append(f.replace('.h0.','.h1.'))

ds = xr.open_dataset(train_feature_files[0])
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
training_features = functions.loadData(train_feature_files[0:n_train],features)
training_labels = functions.loadData(train_label_files[0:n_train],[label])
testing_features = functions.loadData(test_feature_files,features)
testing_labels, PS, time = functions.loadData(test_label_files,[label],get_PS_times=True)
PS = PS[-312:]
time = time[-312:]

# Arange data into single array
training_features = functions.arangeArrays(training_features,features)[53:] # Disregard first year (spin-up data)
training_labels = functions.arangeArrays(training_labels,[label])[53:]
testing_features = functions.arangeArrays(testing_features,features)[-312:] # Take final six years
testing_labels = functions.arangeArrays(testing_labels,[label])[-312:]

if inspect_data:
    D = functions.fill_dict(D,testing_features,time,lev,lat,lon,features,'raw')
    D = functions.fill_dict(D,testing_labels,time,lev,lat,lon,[label],'raw')

print(training_features.shape)
print(training_labels.shape)
print(testing_features.shape)
print(testing_labels.shape)

# Roll data axes
training_features = functions.roll_axes(training_features)
training_labels = functions.roll_axes(training_labels)
testing_features = functions.roll_axes(testing_features)
testing_labels = functions.roll_axes(testing_labels)

print(training_features.shape)
print(training_labels.shape)
print(testing_features.shape)
print(testing_labels.shape)

if not raw:

    # Subtract the mean and divide by the standard deviation
    training_features = functions.scale_ui(training_features, l_weights=l_weights, v_weights=v_weights, lat=True)
    training_labels = functions.scale_ui(training_labels, l_weights=l_weights, v_weights=v_weights)
    testing_features = functions.scale_ui(testing_features, l_weights=l_weights, v_weights=v_weights, lat=True)
    testing_labels, mean, std = functions.scale_ui(testing_labels, l_weights=l_weights, v_weights=v_weights, 
                                                   test=True) 

    if inspect_data:
        D = functions.fill_dict(D,np.rollaxis(testing_features,3,1),time,lev,lat,lon,features,'ui')
        D = functions.fill_dict(D,np.rollaxis(testing_labels,3,1),time,lev,lat,lon,[label],'ui')

    # Scale data to range [-1,1] 
    training_features = functions.scale_ab(training_features,lat=LAT)
    training_labels = functions.scale_ab(training_labels,a=0)
    testing_features = functions.scale_ab(testing_features,lat=LAT)
    testing_labels, dmin, dmax = functions.scale_ab(testing_labels,a=0,test=True)

if inspect_data:
    D = functions.fill_dict(D,np.rollaxis(testing_features,3,1),time,lev,lat,lon,features,'1_1')
    D = functions.fill_dict(D,np.rollaxis(testing_labels,3,1),time,lev,lat,lon,[label],'1_1')
    
if inspect_data:
    xr.Dataset(D).to_netcdf(out_dir+'inspect.nc')

if chop_levs > 0: # Some fields are zero top of the atmosphere, if so we can remove these levels if we want
    print("chopping")
    training_features = training_features[:,:,:,chop_levs:,:] 
    training_labels = training_labels[:,:,:,chop_levs:,:]
    testing_features = testing_features[:,:,:,chop_levs:,:]
    testing_labels = testing_labels[:,:,:,chop_levs:,:]
    print(training_features.shape)
    print(training_labels.shape)
    print(testing_features.shape)
    print(testing_labels.shape)

# Reshape data
training_features = functions.reshape(training_features)
training_labels = functions.reshape(training_labels)
testing_features = functions.reshape(testing_features)
testing_labels, testing_label_shape = functions.reshape(testing_labels, test=True)

print(training_features.shape)
print(training_labels.shape)
print(testing_features.shape)
print(testing_labels.shape)

# Save array(s) to disk
np.savez(out_dir+data_filename,
         train_x = training_features,
         train_y = training_labels,
         test_x = testing_features,
         test_y = testing_labels,
         test_y_shape = testing_label_shape,
         PS = PS,
         time = time)

