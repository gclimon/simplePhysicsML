import functions
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from keras.losses import MeanSquaredError


### Initialize ###
run_type = 'TJBM' 
res = '19'
features = ['T','P','Q','LHFLX','SHFLX','lat']
label = 'PRECC'
data_dir = '/glade/work/glimon/simplePhysML/'+res+'/data/'
data_filename = (run_type+'_'+label+'.npz')
data = np.load(data_dir+data_filename,allow_pickle=True)

output_dir = '/glade/work/glimon/simplePhysML/'+res+'/'+run_type+'/'+label+'/rf/'

best = {'Trial-ID': 1, 'Iteration': 1, 'max_depth': 37, 'min_leafs': 11, 'min_split': 2, 'n_trees': 132, 'Objective': 3.2736434984125787e-17}

best['n_train'] = 20000000
for i in best:
    print(i,best[i])

shpY = data['test_y'].shape
shpX = data['train_x'].shape
predict_y = np.empty(shpY)

train_x = np.reshape(data['train_x'],(shpX[0],shpX[1]*shpX[2]))
test_x = np.reshape(data['test_x'],(shpY[0],shpX[1]*shpX[2]))

X,y = functions.unison_shuffled_copies(train_x,
                                       data['train_y'])
print(X.shape)
print(y.shape)

# for lev in range(predict_y.shape[1]):
model =  RandomForestRegressor(n_estimators=best['n_trees'],
                               max_depth=best['max_depth'],
                               min_samples_split=best['min_split'],
                               min_samples_leaf=best['min_leafs'],
                               n_jobs=32,random_state=0)
model.fit(X[0:best['n_train'],:],y[0:best['n_train'],0])

predict_y[:,0,0] = model.predict(test_x)
print("predicted model")
print("")

print("Test metrics:")
mse = MeanSquaredError()
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
