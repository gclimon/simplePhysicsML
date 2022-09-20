import numpy as np
import xarray as xr        

def setup_output_dict(f,PS,time):
    """
    Function that initializes a dictionary that will be used to simply create the output netcdf file using xarray.
    Args:
        f (str): One of the original CAM output files for the labels; used to get field attributes from.
        PS (ndarray): array of surface pressure field from the feature data (could be replaced now that label files
            contain the field.
        time (ndarray): array with the time field, was originally used because of coarsening, could be removed.
    """
    dataset = xr.open_dataset(f)
    D = {}
    D['PS'] = xr.DataArray(data=PS[:,0,:,:],dims=dataset['PS'].dims,
                           attrs=dataset['PS'].attrs,
                           coords={'time':time,'lat':dataset['PS'].coords['lat'],
                                   'lon':dataset['PS'].coords['lon']})
    D['lev'] = xr.DataArray(data=dataset['lev'].values,
                            attrs=dataset['lev'].attrs,
                            coords=dataset['lev'].coords)
    D['lat'] = xr.DataArray(data=dataset['lat'].values,
                            attrs=dataset['lat'].attrs,
                            coords=dataset['lat'].coords)
    D['lon'] = xr.DataArray(data=dataset['lon'].values,
                            attrs=dataset['lon'].attrs,
                            coords=dataset['lon'].coords)
    D['hyam'] = xr.DataArray(data=dataset['hyam'].values,
                             attrs=dataset['hyam'].attrs,
                             dims=dataset['hyam'].dims,
                             coords=dataset['hyam'].coords)
    D['hybm'] = xr.DataArray(data=dataset['hybm'].values,
                             attrs=dataset['hybm'].attrs,
                             dims=dataset['hybm'].dims,
                             coords=dataset['hybm'].coords)
    D['hyai'] = xr.DataArray(data=dataset['hyai'].values,
                             attrs=dataset['hyai'].attrs,
                             dims=dataset['hyai'].dims,
                             coords=dataset['hyai'].coords)
    D['hybi'] = xr.DataArray(data=dataset['hybi'].values,
                             attrs=dataset['hybi'].attrs,
                             dims=dataset['hybi'].dims,
                             coords=dataset['hybi'].coords)
    return D

def export_data(f, data_dict, label, output_dict, time, output_dir, outfile):
    """
    Function that adds the test tendancies, ML predicted tendancies, and difference to the xarray dataset and 
    writes out the netcdf file. (there may be a more streamlined approach to this...)
    Args:
        f (str): One of the original CAM output files for the labels; used to get field attributes from.
        data_dict (dict): dictionary containing the output data fields
        label (str): the chosen label (tendancy) name
        output_dict (dict): the output dict for xarray that was set up with 'setup_output_dict'
        time (ndarray): array with the time field, was originally used because of coarsening, could be removed.
        output_dir (str): location for file to be placed.
        outfile (str): output netcdf filename.
    """
    dataset = xr.open_dataset(f.replace('.h0.','.h1.'))
    for key in data_dict.keys():
        output_dict[key] = xr.DataArray(data=data_dict[key][:,:,:,:],
                                      attrs=dataset[label].attrs,
                                      dims=dataset[label].dims,
                                      coords={'time':time,'lev':
                                              dataset['lev'].values,
                                              'lat':dataset[label].coords['lat'],
                                              'lon':dataset[label].coords['lon']})
    DS = xr.Dataset(output_dict)
    DS.to_netcdf(output_dir+outfile)

def unroll(data_array):
    """Unrolls the array to correct shape"""
    return np.rollaxis(data_array,3,1)

def unreshape(data_array,shp):
    """Reverts shape from 'vectorized' to 4D"""
    return data_array.reshape((shp[0],shp[1],shp[2],shp[3],shp[4]))

def unscale_ui(data_array, mean, std):
    """Unscales the data from unitarily invarient scaling"""
    return data_array * std + mean

def unscale_ab(data_array, dmin, dmax, a, b):
    """Unscales the data from unitarily invarient scaling"""
    return (data_array - a) / (b - a) * (dmax - dmin) + dmin

def unscale(data_array, scaler):
    """Unscales the data from min-max scaling"""
    return scaler[0].inverse_transform(data_array)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
