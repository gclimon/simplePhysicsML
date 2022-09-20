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
features = ['T','P','Q','RELHUM','LHFLX','SHFLX']
label = 'PTEQ'
data_dir = '/glade/work/glimon/simplePhysML/'+res+'/data/'
data_filename = (run_type+'_'+label+'.npz')
data = np.load(data_dir+data_filename,allow_pickle=True)


output_dir = '/glade/work/glimon/simplePhysML/'+res+'/'+run_type+'/'+label+'/rf/'

load = False

best = {'n_trees': 83, 'n_train': 24166344, 'min_split': 45, 'min_leafs': 13, 'max_depth': 32}

for i in best:
    print(i,best[i])

trx = data['train_x'][0:20000000,6:,:]
shp = trx.shape
X,y = functions.unison_shuffled_copies(
    trx.reshape(shp[0],shp[1]*shp[2])[:],
    data['train_y'][0:20000000,6:,0])

tex = data['test_x'][:,6:,:]
shp1 = tex.shape
tX = tex.reshape(shp1[0],shp1[1]*shp1[2])
ty = data['test_y'][:,6:,0]

model = RandomForestRegressor(n_estimators=best['n_trees'],
                              max_depth=best['max_depth'],
                              min_samples_split=best['min_split'],
                              min_samples_leaf=best['min_leafs'],
                              n_jobs=32,random_state=0)
model.fit(X,y)

with open(output_dir+'RF.pkl', 'wb') as model_file:
  pickle.dump(model, model_file)

py = model.predict(tX)
print("predicted model")
print("")

predict_y = np.zeros(data['test_y'].shape)
predict_y[:,6:,0] = py

print("Test metrics:")
mse = MeanSquaredError()
print("MSE:",mse(ty,py).numpy())

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
