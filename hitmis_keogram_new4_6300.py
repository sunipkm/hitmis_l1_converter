# %% Imports
import datetime as dt
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
from BaselineRemoval import BaselineRemoval
from scipy.optimize import curve_fit
from pysolar import solar
import pytz
from matplotlib.pyplot import cm
import uncertainties as un
import uncertainties.unumpy as unp

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
import matplotlib
rc('font',**{'family':'serif','serif':['Times']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)

#loc 42.64965968479334, -71.31677181842542
# %% Line extraction functions
def lgauss(x, *params):
    y = np.zeros_like(np.asarray(x, dtype = float))
    if len(params) == 1:
        params = params[0]
    # print(params)
    for i in range(5): # 5 degree polynomial
        # print((params[i] * (x**i)).shape)
        y += float(params[i]) * (x**i)
    # print(y)
    for i in range(5, len(params), 3):
        ctr = params[i]
        amp = params[i + 1]
        wid = params[i + 2]
        dy = (amp * np.exp(-((x - ctr)/wid)**2))
        y += dy
    return y

def lback(x, *params):
    y = np.zeros_like(x)
    if len(params) == 1:
        params = params[0]
    for i in range(5):
        y += float(params[i]) * (x ** i)
    return y

def lfgauss(x, id, *params):
    y = np.zeros_like(x)
    if len(params) == 1:
        params = params[0]
    for i in range(5):
        y += float(params[i]) * (x ** i)
    base = 5 + (3 * id)
    ctr = params[base]
    amp = params[base + 1]
    wid = params[base + 2]
    y += (amp * np.exp(-((x - ctr)/wid)**2))
    return y

# %%
def conversion_fcn(didx, xax, img, std, guess, ctrs_6300, amps_6300, wids_6300, sctrs_6300, samps_6300, swids_6300, ctrs_6305, amps_6305, wids_6305, sctrs_6305, samps_6305, swids_6305, ctrs_6306, amps_6306, wids_6306, sctrs_6306, samps_6306, swids_6306):
    for lidx in range(img.shape[0]): # for each look angle
        try:
            popt, pcov = curve_fit(lgauss, xax, np.asarray(img[lidx, :], dtype=float), p0 = guess, sigma=np.asarray(std[lidx, :], dtype=float), check_finite=False, bounds=((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -0.1, -0.01, 0, 0.49, -0.01, 0, 0.6, -0.01, 0), (np.inf, np.inf, np.inf, np.inf, np.inf, 0.1, np.inf, 1, 0.55, np.inf, 0.1, 0.7, np.inf, 0.1)))
            plt.show()
            if -0.3 < popt[5] < 0.3:
                ctrs_6300[didx, lidx] = (popt[5])
                amps_6300[didx, lidx] = (popt[6])
                wids_6300[didx, lidx] = (popt[7])
                sctrs_6300[didx, lidx] = (pcov[5, 5])
                samps_6300[didx, lidx] = (pcov[6, 6])
                swids_6300[didx, lidx] = (pcov[7, 7])
                
            if 0.4 < popt[8] < 0.6:
                ctrs_6305[didx, lidx] = (popt[8])
                amps_6305[didx, lidx] = (popt[9])
                wids_6305[didx, lidx] = (popt[10])
                sctrs_6305[didx, lidx] = (pcov[8, 8])
                samps_6305[didx, lidx] = (pcov[9, 9])
                swids_6305[didx, lidx] = (pcov[10, 10])
            
            if 0.5 < popt[11] < 0.7:
                ctrs_6306[didx, lidx] = (popt[11])
                amps_6306[didx, lidx] = (popt[12])
                wids_6306[didx, lidx] = (popt[13])
                sctrs_6306[didx, lidx] = (pcov[11, 11])
                samps_6306[didx, lidx] = (pcov[12, 12])
                swids_6306[didx, lidx] = (pcov[13, 13])
            # guess = popt.tolist()
            # print(dt.datetime.fromtimestamp(float(valid_ts[didx])), float(img['height'][lidx]), un.ufloat(ctrs_6300[didx, lidx], sctrs_6300[didx, lidx]) + 630, un.ufloat(amps_6300[didx, lidx], samps_6300[didx, lidx]), un.ufloat(wids_6300[didx, lidx], swids_6300[didx, lidx]))
        except Exception as e:
            pass
# %%
plotdir = 'keograms'
try:
    os.mkdir(plotdir)
except FileExistsError:
    pass
# %%
def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0)**2/(2 * sigma ** 2))
# %%
def get_wl(name):
    name = name.split('.nc')[0]
    wl = name.split('_')[-1]
    if len(wl) != 4:
        raise ValueError('%s not valid wavelength: has length %d'%(wl, len(wl)))
    return int(wl)

def get_date(name):
    name = name.split('.nc')[0]
    date = name.split('_')[-2]
    if len(date) != 8:
        raise ValueError('%s not valid date: has length %d'%(date, len(date)))
    return date
# %%
files = glob.glob('%s/../hitmis_locsst/*.nc'%(os.getcwd()))
dates = [get_date(f) for f in files]
wls = [get_wl(f) for f in files]
dates = list(set(dates))
wls = list(set(wls))
dates.sort()
wls.sort()
print(dates)
print(wls)
# %%
tstamp_vals = []
keo_6300 = []
std_6300 = []
keo_6305 = []
std_6305 = []
keo_6306 = []
std_6306 = []
for date in dates:
    # if date not in ['20220126',\
    #                 '20220209',\
    #                 '20220215',\
    #                 '20220218',\
    #                 '20220219',\
    #                 '20220226',\
    #                 '20220303',\
    #                 '20220404']:
    #     continue
    files = glob.glob('%s/../hitmis_locsst/hitmis_dsamp_%s_6300.nc'%(os.getcwd(), date))
    start = dt.datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 20, 30, 0) # 7 pm
    print(start)
    end = dt.datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 3, 30, 0) + dt.timedelta(days = 1)
    ds = xr.load_dataset(files[0])
    slc = slice(start.timestamp(), end.timestamp())
    if len(ds.loc[dict(tstamp=slc)]['tstamp']) == 0:
        print('No valid exposures in %s'%(files[0]))
        continue
    valid_ts = ds.loc[dict(tstamp=slc)]['tstamp']
    data_date = dt.datetime.fromtimestamp(float(valid_ts[0])).strftime('%Y-%m-%d')
    data_start = dt.datetime.fromtimestamp(float(valid_ts[0])).strftime('%H:%M:%S')
    data_end = dt.datetime.fromtimestamp(float(valid_ts[-1])).strftime('%H:%M:%S')
    f = files[0]
    w = get_wl(f) * 0.1
    today_str = get_date(f)
    ds = xr.load_dataset(f)
    imgs = ds.loc[dict(tstamp=slc)]['imgs']
    stds = ds.loc[dict(tstamp=slc)]['stds']
    ts = ds.loc[dict(tstamp=slc)]['tstamp']
    wl = np.asarray(ds['wl'])
    print('Wl sel:', wl.min(), wl.max())
    wt = np.asarray(imgs[0] > 0, dtype = float)
    for k in range(wt.shape[1]):
        if np.sum(wt[:,k]) == 0:
            wt[0, k] = 1
    wts = [wt for _ in range(len(ts))]
    wts = np.asarray(wts)
    wslc = slice(631.2, 0)
    
    iguess = [0.4, 0, 0, 0, 0,
          0, 0.5, 0.1,
        0.51, 0.15, 0.05,
        0.65, 0.1, 0.05]

    for wl in [6300, 6305, 6306]:
        locals()['ctrs_%d'%(wl)] = np.zeros((imgs.shape[0], imgs.shape[1]), dtype = float)
        locals()['amps_%d'%(wl)] = np.zeros((imgs.shape[0], imgs.shape[1]), dtype = float)
        locals()['wids_%d'%(wl)] = np.ones((imgs.shape[0], imgs.shape[1]), dtype = float)
        locals()['sctrs_%d'%(wl)] = np.zeros((imgs.shape[0], imgs.shape[1]), dtype = float)
        locals()['samps_%d'%(wl)] = np.zeros((imgs.shape[0], imgs.shape[1]), dtype = float)
        locals()['swids_%d'%(wl)] = np.zeros((imgs.shape[0], imgs.shape[1]), dtype = float)

    for didx in range(len(imgs)): # for each timestamp
        img = imgs[didx].loc[dict(wl=wslc)] # image
        std = stds[didx].loc[dict(wl=wslc)] # std
        xax = img['wl'] - 630 # wavelength axes, with offset
        guess = iguess
        update_guess = True
        if update_guess:
            tstart = dt.datetime.now()
            try:
                vimg = np.asarray(np.average(img, axis = 0))
                vstd = np.asarray(np.average(std, axis = 0))
                popt, pcov = curve_fit(lgauss, xax, np.asarray(vimg, dtype=float), p0 = guess, sigma=vstd, check_finite=False, bounds=((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -0.1, -0.01, 0, 0.49, -0.01, 0, 0.6, -0.01, 0), (np.inf, np.inf, np.inf, np.inf, np.inf, 0.1, np.inf, 1, 0.55, np.inf, 0.1, 0.7, np.inf, 0.1)))
                if -0.3 < popt[5] < 0.3:
                    guess = popt.tolist()
            except Exception as e:
                print('VAVG update: %s. %d/%d'%(str(e), didx, len(imgs)))
            tend = dt.datetime.now()
            print('Guess estimation for [%d/%d] took %.6f s'%(didx + 1, len(imgs), (tend - tstart).total_seconds()))
        print('Starting conversion [%d/%d]'%(didx + 1, len(imgs)))
        tstart = dt.datetime.now()
        conversion_fcn(didx, xax, img, std, guess, ctrs_6300, amps_6300, wids_6300, sctrs_6300, samps_6300, swids_6300, ctrs_6305, amps_6305, wids_6305, sctrs_6305, samps_6305, swids_6305, ctrs_6306, amps_6306, wids_6306, sctrs_6306, samps_6306, swids_6306)
        # for lidx in range(img.shape[0]): # for each look angle
        #     try:
        #         popt, pcov = curve_fit(lgauss, xax, np.asarray(img[lidx, :], dtype=float), p0 = guess, sigma=np.asarray(std[lidx, :], dtype=float), check_finite=False, bounds=((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -0.1, -0.01, 0, 0.49, -0.01, 0, 0.6, -0.01, 0), (np.inf, np.inf, np.inf, np.inf, np.inf, 0.1, np.inf, 1, 0.55, np.inf, 0.1, 0.7, np.inf, 0.1)))
        #         plt.show()
        #         if -0.3 < popt[5] < 0.3:
        #             ctrs_6300[didx, lidx] = (popt[5])
        #             amps_6300[didx, lidx] = (popt[6])
        #             wids_6300[didx, lidx] = (popt[7])
        #             sctrs_6300[didx, lidx] = (pcov[5, 5])
        #             samps_6300[didx, lidx] = (pcov[6, 6])
        #             swids_6300[didx, lidx] = (pcov[7, 7])
                    
        #         if 0.4 < popt[8] < 0.6:
        #             ctrs_6305[didx, lidx] = (popt[8])
        #             amps_6305[didx, lidx] = (popt[9])
        #             wids_6305[didx, lidx] = (popt[10])
        #             sctrs_6305[didx, lidx] = (pcov[8, 8])
        #             samps_6305[didx, lidx] = (pcov[9, 9])
        #             swids_6305[didx, lidx] = (pcov[10, 10])
        #         # guess = popt.tolist()
        #         # print(dt.datetime.fromtimestamp(float(valid_ts[didx])), float(img['height'][lidx]), un.ufloat(ctrs_6300[didx, lidx], sctrs_6300[didx, lidx]) + 630, un.ufloat(amps_6300[didx, lidx], samps_6300[didx, lidx]), un.ufloat(wids_6300[didx, lidx], swids_6300[didx, lidx]))
        #     except Exception as e:
        #         pass
        tend = dt.datetime.now()
        print('[%d/%d] took %.6f s'%(didx + 1, len(imgs), (tend - tstart).total_seconds()))

    wid_data_6300 = unp.uarray(wids_6300, swids_6300)
    amp_data_6300 = unp.uarray(amps_6300, samps_6300)

    keo_data_6300 = amp_data_6300 * unp.sqrt(np.pi / wid_data_6300)

    wid_data_6305 = unp.uarray(wids_6305, swids_6305)
    amp_data_6305 = unp.uarray(amps_6305, samps_6305)

    keo_data_6305 = amp_data_6305 * unp.sqrt(np.pi / wid_data_6305)
    
    wid_data_6306 = unp.uarray(wids_6306, swids_6306)
    amp_data_6306 = unp.uarray(amps_6306, samps_6306)

    keo_data_6306 = amp_data_6306 * unp.sqrt(np.pi / wid_data_6306)
            
    tstamp_vals.append(valid_ts)
    imgs = xr.DataArray(unp.nominal_values(keo_data_6300), coords=[valid_ts['tstamp'], imgs['height']], dims = ['tstamp', 'look_angle'])
    # print(imgs_)
    keo_6300.append(imgs)
    simgs = xr.DataArray(unp.std_devs(keo_data_6300), coords=[valid_ts['tstamp'], imgs['look_angle']], dims = ['tstamp', 'look_angle'])
    std_6300.append(simgs)

    enable_plot = True

    if enable_plot:
        fig, ax = plt.subplots(2, 1, figsize = (6, 4.8), sharex = True, tight_layout = True)
        fig.set_dpi(300)
        matplotlib.rcParams.update({'font.size': 10})
        matplotlib.rcParams.update({'axes.titlesize': 10})
        matplotlib.rcParams.update({'axes.labelsize': 10})
        fig.suptitle('US/East %s (%s - %s) | %.1f nm'%(data_date, data_start, data_end, 630))
        cax = []
        for ax_ in ax:
            div = make_axes_locatable(ax_)
            cax.append(div.append_axes('right', size = '5%', pad = 0.05))
        im = ax[0].imshow(imgs.transpose(), aspect = 'auto', extent = (0, (float(ts[-1]) - float(ts[0])) / 60, imgs['look_angle'][-1], imgs['look_angle'][0]), cmap = 'bone')
        fig.colorbar(im, cax = cax[0], label = 'Signal (Counts/s)')
        im = ax[1].imshow(simgs.transpose(), aspect = 'auto', extent = (0, (float(ts[-1]) - float(ts[0])) / 60, imgs['look_angle'][-1], imgs['look_angle'][0]), cmap = 'bone')
        fig.colorbar(im, cax = cax[1], label = 'Noise (Counts/s)')
        # fig.colorbar(im, location = 'top', label = 'Pixel count/s')
        ax[-1].set_xlabel('Time Offset (minutes)')
        for _ax in ax:
            _ax.set_ylabel('Look Angle')
        # ax2.set_ylim(xmin, xmax)
        # plt.savefig('%s/pixis_night_%d_%s.pdf'%(plotdir, w * 10, data_date.replace('-', '')))
        plt.show()

    imgs = xr.DataArray(unp.nominal_values(keo_data_6305), coords=[valid_ts['tstamp'], imgs['look_angle']], dims = ['tstamp', 'look_angle'])
    # print(imgs_)
    keo_6305.append(imgs)
    simgs = xr.DataArray(unp.std_devs(keo_data_6305), coords=[valid_ts['tstamp'], imgs['look_angle']], dims = ['tstamp', 'look_angle'])
    std_6305.append(simgs)

    if enable_plot:
        fig, ax = plt.subplots(2, 1, figsize = (6, 4.8), sharex = True, tight_layout = True)
        fig.set_dpi(300)
        matplotlib.rcParams.update({'font.size': 10})
        matplotlib.rcParams.update({'axes.titlesize': 10})
        matplotlib.rcParams.update({'axes.labelsize': 10})
        fig.suptitle('US/East %s (%s - %s) | %.1f nm'%(data_date, data_start, data_end, 630.5))
        cax = []
        for ax_ in ax:
            div = make_axes_locatable(ax_)
            cax.append(div.append_axes('right', size = '5%', pad = 0.05))
        im = ax[0].imshow(imgs.transpose(), aspect = 'auto', extent = (0, (float(ts[-1]) - float(ts[0])) / 60, imgs['look_angle'][-1], imgs['look_angle'][0]), cmap = 'bone')
        fig.colorbar(im, cax = cax[0], label = 'Signal (Counts/s)')
        im = ax[1].imshow(simgs.transpose(), aspect = 'auto', extent = (0, (float(ts[-1]) - float(ts[0])) / 60, imgs['look_angle'][-1], imgs['look_angle'][0]), cmap = 'bone')
        fig.colorbar(im, cax = cax[1], label = 'Noise (Counts/s)')
        ax[-1].set_xlabel('Time Offset (minutes)')
        for _ax in ax:
            _ax.set_ylabel('Look Angle')
        # ax2.set_ylim(xmin, xmax)
        # plt.savefig('%s/pixis_night_%d_%s.pdf'%(plotdir, w * 10, data_date.replace('-', '')))
        plt.show()

    imgs = xr.DataArray(unp.nominal_values(keo_data_6306), coords=[valid_ts['tstamp'], imgs['look_angle']], dims = ['tstamp', 'look_angle'])
    # print(imgs_)
    keo_6306.append(imgs)
    simgs = xr.DataArray(unp.std_devs(keo_data_6306), coords=[valid_ts['tstamp'], imgs['look_angle']], dims = ['tstamp', 'look_angle'])
    std_6306.append(simgs)
# %%
keo_dict = {
    'tstamp': tstamp_vals,
    'keo_6300': keo_6300,
    'std_6300': std_6300,
    'keo_6305': keo_6305,
    'std_6305': std_6305,
    'keo_6306': keo_6306,
    'std_6306': std_6306
}
# %%
import pickle
import lzma
# %%
with lzma.open('keo_dsamp_pixis_6300.xz', 'wb') as f:
    pickle.dump(keo_dict, f)
# %%
