import functions
import numpy as np
import tensorflow as tf
#import keras as ks
import tensorflow_addons as tfa


### Initialize ###
run_type = 'CLDERA' 
opt = 'passive'
lag = 30 * 4
label = 'T_30'
Ntrain = 250 * 4
Ntest = 10 * 4
# Choose attributes
features = ['T','P','U','V','SAI_AOD','lat']
if 'lat' in features:
    LAT = True
else:
    LAT = False
data_dir = '/glade/work/glimon/simplePhysML/CLDERA/data/'
data_filename = run_type+'_'+label+'.npz'
output_dir = '/glade/work/glimon/simplePhysML/'+run_type+'/'+label+'/esn/'
load = False
best = {'units':64}
print('initialized')

model = tf.keras.Sequential()
model.add(tfa.layers.ESN(units=best['units']))
model.compile()
print('created model')

data = np.load(data_dir+data_filename)
print('loaded data')

model.fit(data['train_x'],data['train_y'])
print('trained model')

py = model.predict(data['test_x'])
print("predicted model")

predict_y = np.zeros(data['test_y'].shape)
predict_y[:,:,0] = py

print("Test metrics:")
mse = ks.metrics.MeanSquaredError()
print("MSE:",mse(data['test_y'],predict_y).numpy())

np.savez(output_dir+'results.npz',
         predict_y=predict_y,
         test_y=data['test_y'],
         test_y_shape=data['test_y_shape'],
         # dmin = data['dmin'],
         # dmax = data['dmax'],
         # mean = data['mean'],
         # std = data['std'],
         PS=data['PS'],
         time=data['time'])
