import functions
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import sherpa

### Initialize ###
run_type = 'TJ' 
res = '19'
features = ['T','P','Q','RELHUM','LHFLX','SHFLX','lat']
label = 'PTTEND'
data_dir = '/glade/work/glimon/simplePhysML/'+res+'/data/'
data_filename = (run_type+'_'+label+'.npz')
data = np.load(data_dir+data_filename,allow_pickle=True)

output_dir = '/glade/work/glimon/simplePhysML/'+res+'/'+run_type+'/'+label+'/rf/'

params = [
    sherpa.Discrete(name='n_trees', range=[50,500]),
    sherpa.Discrete(name='n_train', range=[500000,25000000]),
    sherpa.Discrete(name='min_split', range=[2,50]),
    sherpa.Discrete(name='min_leafs', range=[2,50]),
    sherpa.Discrete(name='max_depth', range=[2,50])
    ]

lev = 29
alg = sherpa.algorithms.RandomSearch(max_num_trials=25)
study = sherpa.Study(parameters=params,
                     algorithm=alg,
                     lower_is_better=True)
for trial in study:
    X,y = functions.unison_shuffled_copies(data['train_x'][0:trial.parameters['n_train'],lev,:],
                                           data['train_y'][0:trial.parameters['n_train'],lev,0])
    model = RandomForestRegressor(n_estimators=trial.parameters['n_trees'],
                                  max_depth=trial.parameters['max_depth'],
                                  min_samples_split=trial.parameters['min_split'],
                                  min_samples_leaf=trial.parameters['min_leafs'],
                                  n_jobs=8,random_state=0)
    model.fit(X,y)
    vX,vy = functions.unison_shuffled_copies(data['train_x'][-50000:,lev,:],
                                             data['train_y'][-50000:,lev,0])
    valid_y = model.predict(vX)
    score = mean_squared_error(vy,valid_y)
    print("Params:",trial.parameters)
    print("Score:",score)
    print("")
    study.add_observation(trial,iteration=1,objective=score)
    study.finalize(trial)

best = study.get_best_result()
print(best)
