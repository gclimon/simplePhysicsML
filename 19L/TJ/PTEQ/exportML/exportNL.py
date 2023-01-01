import numpy as np
import pickle
import xarray as xr
from sklearn.ensemble import RandomForestRegressor

### Initialize ###
run_type = 'TJ' 
res = '19'
features = ['T','P','Q','RELHUM','LHFLX','SHFLX']
label = 'PTEQ'
output_dir = '/glade/work/glimon/simplePhysML/'+res+'/'+run_type+'/'+label+'/rf/'

with open(output_dir+'RF.pkl', 'rb') as model_file:
  model = pickle.load(model_file)

print("params")
print(model.get_params())
print("")
print("base estimator")
print(model.base_estimator_)
print("")
print("estimators")
print(model.estimators_)
print("")
print("feature_importances")
print(model.feature_importances_)
