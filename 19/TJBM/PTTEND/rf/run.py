import functions
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from keras.losses import MeanSquaredError
import pickle

### Initialize ###
run_type = 'TJBM' 
res = '19'
opt = "_50trees"
features = ['T','P','Q','RELHUM','LHFLX','SHFLX']
# features = ['T','P','Q','LHFLX','SHFLX']
label = 'PTTEND'
data_dir = '/glade/work/glimon/simplePhysML/'+res+'/data/'
# data_filename = (run_type+'_'+label+opt+'.npz')
data_filename = (run_type+'_'+label+'.npz')
data = np.load(data_dir+data_filename,allow_pickle=True)

output_dir = '/glade/scratch/glimon/simplePhysML/'+res+'/'+run_type+'/'+label+'/rf/'

load = False

best = {'n_trees': 269, 'n_train': 21702778, 'min_split': 23, 'min_leafs': 18, 'max_depth': 22}
best['n_trees'] = 50

for i in best:
    print(i,best[i])


shp = data['train_x'].shape
X,y = functions.unison_shuffled_copies(
    data['train_x'].reshape(shp[0],shp[1]*shp[2])[0:15000000],
    data['train_y'][0:15000000,:,0])

shp1 = data['test_x'].shape
tX = data['test_x'].reshape(shp1[0],shp1[1]*shp1[2])
ty = data['test_y'][:,:,0]


if load:
    model = pickle.load(open(output_dir+"RF.pkl", 'rb'))
#    model.partial_fit(X,y)
else:
    model =  RandomForestRegressor(n_estimators=best['n_trees'],
                                   max_depth=best['max_depth'],
                                   min_samples_split=best['min_split'],
                                   min_samples_leaf=best['min_leafs'],
                                   n_jobs=36,random_state=0)
    model.fit(X,y)

    with open(output_dir+'RF'+opt+'.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

py = model.predict(tX)
print("predicted model")
print("")

print("Test metrics:")
mse = MeanSquaredError()
print("MSE:",mse(ty,py).numpy())

predict_y = np.zeros(data['test_y'].shape)
predict_y[:,:,0] = py

np.savez(output_dir+'results'+opt+'.npz',
         predict_y=predict_y,
         test_y=data['test_y'],
         test_y_shape=data['test_y_shape'],
         # dmin = data['dmin'],
         # dmax = data['dmax'],
         # mean = data['mean'],
         # std = data['std'],
         PS=data['PS'],
         time=data['time'])
