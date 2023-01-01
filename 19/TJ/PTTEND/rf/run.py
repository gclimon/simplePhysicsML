import functions
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from keras.losses import MeanSquaredError
import pickle

### Initialize ###
run_type = 'TJ' 
res = '19'
optIN = '_aug'
optOUT = '_1M'
# features = ['T','P','Q','LHFLX','SHFLX']
features = ['T','P','Q','RELHUM','LHFLX','SHFLX']
label = 'PTTEND'
data_dir = '/glade/scratch/glimon/simplePhysML/'+res+'/data/'

data_filename = (run_type+'_'+label+optIN+'.npz')
# data_filename = (run_type+'_'+label+'.npz')
data = np.load(data_dir+data_filename,allow_pickle=True)

output_dir = '/glade/scratch/glimon/simplePhysML/'+res+'/'+run_type+'/'+label+'/rf/'

load = False

best = {'n_trees': 50, 'n_train': 24773032, 'min_split': 20, 'min_leafs': 15, 'max_depth': 30}
# best = {'n_trees': 250, 'n_train': 24773032, 'min_split': 20, 'min_leafs': 15, 'max_depth': 30}
# best = {'Trial-ID': 1, 'Iteration': 1, 'max_depth': 37, 'min_split': 20, 'min_leafs': 46, 'n_trees': 352, 'Objective': 5.179675371221315e-11}

for i in best:
    print(i,best[i])

shp = data['train_x'].shape
print(shp)
X,y = functions.unison_shuffled_copies(
    data['train_x'].reshape(shp[0],shp[1]*shp[2])[0:1000000],
    data['train_y'][0:1000000,:,0])


shp1 = data['test_x'].shape
print(shp1)
tX = data['test_x'].reshape(shp1[0],shp1[1]*shp1[2])
ty = data['test_y'][:,:,0]



model =  RandomForestRegressor(n_estimators=best['n_trees'],
                               max_depth=best['max_depth'],
                               min_samples_split=best['min_split'],
                               min_samples_leaf=best['min_leafs'],
                               n_jobs=36,random_state=0)
model.fit(X,y)

with open(output_dir+'RF'+optOUT+'.pkl', 'wb') as model_file:
  pickle.dump(model, model_file)

py = model.predict(tX)
print("predicted model")
print("")

print("Test metrics:")
mse = MeanSquaredError()
print("MSE:",mse(ty,py).numpy())

predict_y = np.zeros(data['test_y'].shape)
predict_y[:,:,0] = py

np.savez(output_dir+'results'+optOUT+'.npz',
         predict_y=predict_y,
         test_y=data['test_y'],
         test_y_shape=data['test_y_shape'],
         PS=data['PS'],
         time=data['time'])
