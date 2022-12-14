import functions
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from keras.losses import MeanSquaredError
import pickle

### Initialize ###
run_type = 'HS' 
res = '19'
features = ['T','P','lat']
label = 'PTTEND'
data_dir = '/glade/work/glimon/simplePhysML/'+res+'/data/'
data_filename = (run_type+'_'+label+'.npz')
data = np.load(data_dir+data_filename,allow_pickle=True)

output_dir = '/glade/work/glimon/simplePhysML/'+res+'/'+run_type+'/'+label+'/rf/'

load = False

best = {'Trial-ID': 12, 'Iteration': 1, 'max_depth': 39, 'min_leafs': 6, 'min_split': 17, 'n_trees': 157, 'Objective': 1.575828112756301e-13}

for i in best:
    print(i,best[i])

shp = data['train_x'].shape
print(shp)
train_x,train_y = functions.unison_shuffled_copies(
    data['train_x'].reshape(shp[0],shp[1]*shp[2])[:],
    data['train_y'][:,:,0])
X = train_x
y = train_y


shp1 = data['test_x'].shape
tX = data['test_x'].reshape(shp1[0],shp1[1]*shp1[2])
ty = data['test_y'][:,:,0]



model =  RandomForestRegressor(n_estimators=best['n_trees'],
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

print("Test metrics:")
mse = MeanSquaredError()
print("MSE:",mse(ty,py).numpy())

predict_y = np.zeros(data['test_y'].shape)
predict_y[:,:,0] = py

np.savez(output_dir+'results.npz',
         predict_y=predict_y,
         test_y=data['test_y'],
         test_y_shape=data['test_y_shape'],
         PS=data['PS'],
         time=data['time'])
