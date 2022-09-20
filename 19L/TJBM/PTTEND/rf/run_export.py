import numpy as np
import xarray as xr
import functions

run_type = 'TJBM' 
label = 'PTTEND'
res = '19'

test_feature_file = '/glade/work/glimon/ml_data/'+run_type'_'+res+'/'+run_type+'_'+res+'.cam.h0.0050-11-12-00000.nc'
output_dir = '/glade/work/glimon/ml_climate/'+res+'/'+run_type+'/'+label+'/rf/'
data = np.load(output_dir+'results.npz',allow_pickle=True)

### 'Unprocess' the output ###
test_y = functions.unreshape(data['test_y'],data['test_y_shape'])
test_y = functions.unroll(test_y)
# test_y = functions.unscale_ab(test_y, data['dmin'], data['dmax'], -1., 1.)
# test_y = functions.unscale_ui(test_y,data['mean'],data['std'])

predict_y = functions.unreshape(data['predict_y'],data['test_y_shape'])
predict_y = functions.unroll(predict_y)
# predict_y = functions.unscale_ab(predict_y, data['dmin'], data['dmax'], -1., 1.)
# predict_y = functions.unscale_ui(predict_y,data['mean'],data['std'])

difference = predict_y[:,:,:,:,0] - test_y[:,:,:,:,0]

data_dict = {label:test_y[:,:,:,:,0],
             label+'_ML_predicted':predict_y[:,:,:,:,0],
             label+'_difference':difference}

### Output the results ###
output_dict = functions.setup_output_dict(test_feature_file,data['PS'],data['time'])

functions.export_data(test_feature_file, data_dict, label, 
                      output_dict, data['time'], output_dir, 'results.nc')
