# %%
import xarray as xr
import numpy as np
import os
import sys
import glob
# %%
flist = glob.glob('/home/locsst/codes/hitmis_l1_converter/hitmis_night*.nc')
def getWave(f):
    return int(f.split('hitmis_night_')[1].split('.nc')[0])
flist.sort(key=getWave)
print(flist)
# %%
def filter_data(data):
    freq_low = 20.5 # minutes
    freq_high = 28.6 # minutes
    img_cadence = 126.91376056413695 # seconds
    n = data.shape[0]
    freq = np.fft.fftfreq(n, img_cadence / 60)
    fdata = np.fft.fft(data, axis = 0)
    idx1 = np.where(((1/freq) > freq_low) & ((1/freq) < freq_high))
    idx2 = np.where(((1/freq) > -freq_high) & ((1/freq) < -freq_low))
    fdata[idx1] = 0
    fdata[idx2] = 0 
    ndata = np.fft.ifft(fdata, axis = 0).real
    return ndata
# %%
encoding = {'imgs': {'dtype': float, 'zlib': True},
            'exposure': {'dtype': float, 'zlib': True}}
for f in flist:
    d = xr.open_dataset(f)
    wl = d.attrs['wl']
    print('Working on %.1f nm...'%(wl))
    sys.stdout.flush()
    imgs = filter_data(d['imgs'])
    ts = np.asarray(d['tstamp'])
    exposure = np.asarray(d['exposure'])
    dataset = xr.Dataset(
        data_vars = dict(
            imgs=(['tstamp', 'height', 'wl'], imgs),
            exposure=(['tstamp'], exposure)
        ),
        coords = dict(tstamp=ts),
        attrs = dict(wl=wl)
    )
    fname = 'hitmis_night_l2_%04d.nc'%(wl * 10)
    print('Saving file %s...'%(fname), end = '')
    sys.stdout.flush()
    dataset.to_netcdf(fname, encoding = encoding)
    print("Done.")
# %%
