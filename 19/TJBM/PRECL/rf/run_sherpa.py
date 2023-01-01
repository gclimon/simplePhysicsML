from joblib import parallel_backend
import functions
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import sherpa

### Initialize ###
run_type = 'TJBM' 
res = '19'
features = ['T','P','Q','RELHUM','LHFLX','SHFLX','lat']
label = 'PRECL'
data_dir = '/glade/work/glimon/simplePhysML/'+res+'/data/'
data_filename = (run_type+'_'+label+'.npz')
data = np.load(data_dir+data_filename,allow_pickle=True)

output_dir = '/glade/work/glimon/simplePhysML/'+res+'/'+run_type+'/'+label+'/rf/'

params = [
    sherpa.Discrete(name='n_trees', range=[50,500]),
    sherpa.Discrete(name='min_split', range=[2,50]),
    sherpa.Discrete(name='min_leafs', range=[2,50]),
    sherpa.Discrete(name='max_depth', range=[2,50])
    ]

alg = sherpa.algorithms.RandomSearch(max_num_trials=5)
study = sherpa.Study(parameters=params,
                     algorithm=alg,
                     lower_is_better=True)

train_x = np.reshape(data['train_x'],(data['train_x'].shape[0],data['train_x'].shape[1]*data['train_x'].shape[2]))

X,y = functions.unison_shuffled_copies(train_x[0:1000000,:],
                                       data['train_y'][0:1000000,0,0])


with parallel_backend('threading', n_jobs=32):
    for trial in study:
        model = RandomForestRegressor(n_estimators=trial.parameters['n_trees'],
                                      max_depth=trial.parameters['max_depth'],
                                      min_samples_split=trial.parameters['min_split'],
                                      min_samples_leaf=trial.parameters['min_leafs'],
                                      n_jobs=-1,random_state=0)
        model.fit(X,y)
        vX,vy = functions.unison_shuffled_copies(train_x[-50000:,:],
                                             data['train_y'][-50000:,0,0])
        valid_y = model.predict(vX)
        score = mean_squared_error(vy,valid_y)
        print("Params:",trial.parameters)
        print(score)
        print("")
        study.add_observation(trial,iteration=1,objective=score)
        study.finalize(trial)
        
best = study.get_best_result()
print(best)

# np.savez(output_dir+'sherpa_results.npz',
#          test_y=data['test_y'],
#          test_x=data['test_x'],
#          train_x=data['train_x'],
#          train_y=data['train_y'],
#          test_y_shape=data['test_y_shape'],
#          dmin = data['dmin'],
#          dmax = data['dmax'],
#          mean = data['mean'],
#          std = data['std'],
#          PS=data['PS'],
#          time=data['time'])
