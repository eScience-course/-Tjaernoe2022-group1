import numpy as np
import xarray as xr

def pnsd_integration(ds, size_thresh, var='dNdlogD', dvar = 'D'):
    '''
    - D is the mean diameter of the size bins
    - time is the time dimension
    - ds is the dataset containing the size distribution
    - pnsd is the variable in which the particle number size distribution is stored (dN/dlogDp)
    - size_thresh is the lower limit of the size bins to be integrated in nanometers
    '''
    Dp = ds[dvar].values
    logDp = np.log10(Dp)
    interval = np.array([logDp[i]-logDp[i-1] for i in range(1,np.size(Dp))])/2
    centers = logDp[:-1]+interval
    centers_bis = np.append(logDp[0]-interval[0], centers)
    centers_bis = np.append(centers_bis, logDp[-1]+interval[-1])
    bound_bin = 10**(centers_bis)#*10**(-9)
    dlogDp = np.array([np.log10(bound_bin[i+1])-np.log10(bound_bin[i]) for i in range(0, bound_bin.shape[0]-1)])
    pnsd_nolog = np.zeros(ds[var].values.shape)
    for i, Dp_i in enumerate(Dp):
        pnsd_nolog[:,i] = ds[var].sel(**{dvar : Dp_i}).values * dlogDp[i]
    ds[f'{var}_unlog'] = (['time', dvar], pnsd_nolog)
    ds['N'+str(np.round(size_thresh, 2))] = ds[f'{var}_unlog'].sel(**{dvar:slice(size_thresh, 1000)}).sum(**{'dim':dvar})