import numpy as np
import xarray as xr
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def loadData(files, variables, get_PS_times=False):
    """
    Function to load in data from NETCDF files. Searches through file for specified variables to extract 
    into a numpy array.
    Args:
        files (list): list of file names/locations
        variables (list): list of variable names to extract (lat should always be last if used!)
        get_PS_times (bool): whether to return surface pressure and array of times.
    Returns:
        data_dict (dict): dictionary containing requested raw data as numpy arrays.
        PS (numpy array): array of surface pressure values, for use when exporting ML data back to
            NETCDF files.
        times (numpy array): array of time values, for use when exporting ML data back to NETCDF files
    Notes:
        If requesting 'lat' as variable, must be last in 'variable' list..
    """
    data_dict = {}
    for jf, f in enumerate(files):
        dataset = xr.open_dataset(f)
        data = {}
        for ivar, var in enumerate(variables):
            if var == 'P':
                data[var] = calcP(dataset['P0'].values,
                                   dataset['PS'].values,
                                   dataset['hyam'].values,
                                   dataset['hybm'].values,
                                   dataset.time.size,
                                   dataset.lat.size,
                                   dataset.lon.size,
                                   dataset.lev.size)
            elif var == 'Qr': # Must be after P, T, Q in variables list
                data[var] = calcQr(data['P'],data['T'],dataset['Q'].values)
            else:
                data[var] = dataset[var].values
            if jf == 0:
                data_dict[var] = data[var]
                PS = np.empty((dataset.time.size,1,dataset.lat.size,dataset.lon.size))
                PS[:,0,:,:] = dataset['PS'].values
                times = dataset['time'].values
            else:
                if var != 'lat':
                    data_dict[var] = np.concatenate((data_dict[var],data[var]))
                ps = np.empty((dataset.time.size,1,dataset.lat.size,dataset.lon.size))
                ps[:,0,:,:] = dataset['PS'].values
                PS = np.concatenate((PS,ps))
                times = np.concatenate((times,dataset['time'].values))
    if get_PS_times:
        return data_dict, PS, times
    else:
        return data_dict

def calcQr(P,T,Q):
    """
    Calculates and returns an array of the relative humidity field.
    """
    eps = 0.622
    T0 = 273.16
    e0 = 610.78
    L = 2.5e6
    Rv = 461.5
    Qs = eps*e0*np.exp(-(L/Rv)*(1/T - 1/T0))/P
    return Q/Qs

def calcP(P0,PS,hyam,hybm,time,lat,lon,lev):
    """
    Calculates and returns an array of the pressure field.
    """
    P_0 = P0 * np.ones([time,lat,lon])
    P = np.empty([time,lev,lat,lon])
    for l in range(lev):
        P[:,l,:,:] = hyam[l]*P_0 + hybm[l]*PS
    return P

def expandLat(lat, shp):
    data = np.empty(shp)
    for t in range(shp[0]):
        for l in range(shp[2]):
            data[t,:,l] = lat
    return data


def extendSurfaceField(data, levs, fill=False):
    shp = data.shape
    new_data = np.zeros((shp[0],levs,shp[1],shp[2]))
    if fill:
        for l in range(levs):
            new_data[:,l,:,:] = data
    else:
        new_data[:,-1,:,:] = data
    return new_data
    
def arangeArrays(data_dict, variables, n_chop=0):
    shp = data_dict[variables[0]].shape
    if 'PREC' in variables[0]:
        new_data = np.empty((shp[0],1,shp[1],shp[2],len(variables)))
        new_data[:,0,:,:,0] = data_dict[variables[0]] 
    else:
        new_data = np.empty((shp[0],shp[1],shp[2],shp[3],len(variables)))
        for ivar, var in enumerate(variables):
            if var=='lat':
                new_data[:,:,:,:,ivar] = extendSurfaceField(
                    expandLat(data_dict[var],[shp[0],shp[2],shp[3]]),shp[1],fill=True)
            elif 'FLX' in var: 
                new_data[:,:,:,:,ivar] = extendSurfaceField(data_dict[var],shp[1])
            # elif var=='PRECL': 
            #     new_data[:,:,:,:,ivar] = extendSurfaceField(data_dict[var],shp[1])
            elif var=='PS':
                new_data[:,:,:,:,ivar] = extendSurfaceField(data_dict[var],shp[1],fill=True)
            else:
                new_data[:,:,:,:,ivar] = data_dict[var]
    if n_chop>0:
        return new_data[:,n_chop:shp[1],:,:,:]
    else:
        return new_data

def stackArrays(data_dict, variables, n_chop=0, single=False):
    """
    Preprocessing function that takes the dictionary of data and stacks individual arrays
    into a single numpy array along the vertical column dimension.
    Args:
    data_dict (dict): dictionary containg raw data as individual arrays (output of loadData)
    variables (list): list of variables to be 'stacked'
    n_chop (int): number of levels to disregard in the case of no data above a certain vertical level.
    Returns:
        data_array (numpy array): single array of all data, various variables are stacked along vertical 
            column (dimension 1)
    """
    for ivar,var in enumerate(variables):
        if 'FLX' in var or 'PREC' in var:
            field_shape = data_dict[var].shape
            a = np.empty([field_shape[0],1,field_shape[1],field_shape[2]])
            a[:,0,:,:] = data_dict[var]
        elif var == 'lat':
            t_shape = data_dict['T'].shape
            a = np.empty([t_shape[0],1,t_shape[2],t_shape[3]])
            for l in range(t_shape[2]):
                a[:,0,l,:] = data_dict[var][l]
        else:
            a = data_dict[var]
        if single:
            nlev = n_chop+1
        else:
            # if a.shape[1]==1:
            #     nlev=2
            # else:
            nlev = a.shape[1]
        if ivar == 0:
            array = a[:,n_chop:nlev,:,:]
        else:
            if var=='lat':
                array = np.concatenate((array,a),axis=1)
            else:
                array = np.concatenate((array,a[:,n_chop:nlev,:,:]),axis=1)
    return array

def fill_dict(d, data, time, lev, lat, lon, labels, step):
    for il,l in enumerate(labels):
        d[l+'_'+step] = xr.DataArray(data[:,:,:,:,il],
                                     coords={'time':time,'lev':lev,'lat':lat,'lon':lon},
                            dims=['time','lev','lat','lon'])
    return d

def roll_axes(data_array, test=False):
    """
    Preprocessing function that rolls the vertical column axis (1) to the final dimension.
    """
    return np.rollaxis(np.rollaxis(data_array,2,1),3,2)

def reshape(data_array, test=False, stack=False, filled=[]):
    """
    Preprocessing function that reshapes the data_array such that its first dimension is now
    a dimension of size TIMExLATxLON.
    """
    shp = data_array.shape
    data_array = data_array.reshape((shp[0]*shp[1]*shp[2],shp[3],shp[4]))
    if stack:
        shp3 = 0
        for i in range(shp[4]):
            if i in filled:
                shp3 += 1
            else:
                shp3 += shp[3]
        new_array = np.empty((shp[0]*shp[1]*shp[2],shp3))
        for i in range(shp[4]):
            if i in filled:
                new_array[:,j] = data_array[:,0,i]
                j += 1
            else:
                j = i*shp[3] + shp[3]
                new_array[:,i*shp[3]:j] = data_array[:,:,i]
    else:
        new_array = data_array
    if test:
        return new_array, shp
    else:
        return new_array

def weighted_avg_std(values, axis, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, axis=axis, weights=weights)
    variance = np.average((values-average)**2, axis=axis, weights=weights)
    return (average, np.sqrt(variance))

def scale_ui(data, fill=[], l_weights=None, test=False):
    """
    Preprocessing function that scales data to become unitarily invariant for machine learning. This simply 
    means we subtract the mean and divide by the standard deviation.
    """
    mean = data.mean(axis=0).mean(axis=1)
    std = data.std(axis=0).std(axis=1)
    if l_weights is not None:
        mean,std = weighted_avg_std(mean, axis=0, weights=l_weights)
    else:
        mean = mean.mean(axis=0)
        std = std.std(axis=0)
    new_data = (data - mean) / std
    if test:                        
        return new_data, mean, std
    else:
        return new_data

def scale_ab(data,fill=[], a=-1.,b=1.,test=False,lat=False):
    """
    Preprocessing function that scales data into the range [-1,1]. This is for machine necesssar ylearning
    to work as many variables have drastically different order of magnitude in their respective ranges.
    """    
    new_data = np.zeros(data.shape)
    dmin = data.min(axis=0).min(axis=0).min(axis=0).min(axis=0)
    dmax = data.max(axis=0).max(axis=0).max(axis=0).max(axis=0)    
    new_data = (data - dmin) / (dmax - dmin) * (b - a) + a
    if test:
        return new_data, dmin, dmax
    else:
        return new_data

