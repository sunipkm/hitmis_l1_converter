# %% Imports
from functools import partial
import gc
import os
import sys
import glob
import astropy.io.fits as pf
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from skimage import transform
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import lzma
import pickle
import datetime
import argparse
import uncertainties as un
from uncertainties import unumpy as unp
from time import perf_counter_ns
# %% Argument Parser
parser = argparse.ArgumentParser(
    description='Convert HiT&MIS L0 data to L1 data, with exposure normalization and timestamp regularization (1 image for every 4 minutes), which separates data by filter region and performs line straightening using ROI listed in accompanying "hitmis_roi.csv" file, and edge lines in "edge_detection" directory. The program will not work without these files present.')
# %% Add arguments
parser.add_argument('rootdir',
                    metavar='rootdir',
                    type=str,
                    help='Root directory containing HiT&MIS data')
parser.add_argument('dest',
                    nargs='?',
                    default=os.getcwd(),
                    help='Root directory where L1 data will be stored')
parser.add_argument('--test',
                    required=False,
                    action='store_true',
                    help='Test line straightening result')
# %% Parse arguments
args = parser.parse_args()

rootdir = args.rootdir

if not os.path.isdir(rootdir):
    print('Specified root directory for L0 data does not exist.')
    sys.exit()

destdir = args.dest

print('Files are being stored in:', end='\n\t')
print(destdir, end='\n\n')
# if destdir is None or not os.path.isdir(destdir):
#     print('Specified destination directory does not exist, output L1 data will be stored in current directory.\n')
#     destdir = './'

testing = args.test
# %% Get all subdirs


def list_all_dirs(root):
    flist = os.listdir(root)
    # print(flist)
    out = []
    subdirFound = False
    for f in flist:
        if os.path.isdir(root + '/' + f):
            subdirFound = True
            out += list_all_dirs(root + '/' + f)
    if not subdirFound:
        out.append(root)
    return out


# %% Load in file list
dirlist = list_all_dirs(rootdir)
# %% Get all files


def getctime(fname):
    words = fname.rstrip('.fit').split('_')
    return int(words[-1])


filelist = []
for d in dirlist:
    if d is not None:
        f = glob.glob(d+'/*.fit')
        f.sort(key=getctime)
        filelist.append(f)

flat_filelist = []
for f in filelist:
    for img in f:
        flat_filelist.append(img)

flist = flat_filelist
flist.sort(key=getctime)
# %% Get timeframe
start_date = datetime.datetime.fromtimestamp(getctime(flist[0])*0.001)
end_date = datetime.datetime.fromtimestamp(getctime(flist[-1])*0.001)
print('First image:', start_date)
print('Last image:', end_date)
print('\n')
# %% Break up into individual days, day is noon to noon
st_date = start_date.date() - datetime.timedelta(days=1)
lst_date = end_date.date() + datetime.timedelta(days=1)
main_flist = {}
all_files = []
print('Dates with data: ', end = '')
data_found = False
first = True
while st_date <= lst_date:
    _st_date = st_date
    start = datetime.datetime(
        st_date.year, st_date.month, st_date.day, 6, 0, 0) # 6 am
    st_date += datetime.timedelta(days=1)
    stop = datetime.datetime(
        st_date.year, st_date.month, st_date.day, 5, 59, 59) # 6 am
    start_ts = start.timestamp() * 1000
    stop_ts = stop.timestamp() * 1000
    valid_files = [f if start_ts <= getctime(
        f) <= stop_ts else '' for f in flist]
    while '' in valid_files:
        valid_files.remove('')
    if len(valid_files) > 0:
        data_found = True
        main_flist[_st_date] = valid_files
        all_files += valid_files
        if first:
            print(_st_date, end = '')
            first = False
        else:
            print(',',_st_date, end = '')
        sys.stdout.flush()
if not data_found:
    print('None')
print('\n')

# %% Define resolution function
resolutions = {}
ifile = open('hitmis_resolution.txt')
for line in ifile:
    words = line.rstrip('\n').split()
    resolutions[float(words[0])] = [float(w) for w in words[1:]]
ifile.close()

# %% Define ROI function
roidict = {}


def get_roi(wl, cond=False):
    global roidict
    if int(wl*10) not in roidict.keys():
        roi = np.loadtxt('hitmis_roi.csv', skiprows=1,
                         delimiter=',').transpose()
        coord = np.where(roi == wl)
        coords = roi[2:, coord[1]]
        try:
            xmin = int(coords[0]) * (2 if cond else 1)
            xmax = int(coords[0] + coords[2]) * (2 if cond else 1)
            ymin = int(coords[1]) * (2 if cond else 1)
            ymax = int(coords[1] + coords[3]) * (2 if cond else 1)
        except Exception:
            return {'xmin': -1, 'ymin': -1, 'xmax': -1, 'ymax': -1}
        roidict[int(wl*10)] = {'xmin': xmin,
                               'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        return {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
    else:
        return roidict[int(wl*10)]
# %% Defines fit lines


def get_lines(wl):
    roi = get_roi(wl)
    data = np.loadtxt('edge_detection/%d_edge.txt' % (wl*10)).transpose()
    # data[0] -= roi['xmin']
    # data[1] -= roi['ymin']
    return data


# %% Wavelengths
wls = [486.1, 557.7, 630.0, 656.3]
# %% Test image
idx = np.random.randint(0, high=len(flist))
img = np.asarray(pf.open(flist[idx])[1].data, dtype=float)
# %% Line straightening functions
pcoeffdict = {}
projdict = {}
coladjdict = {}
transformptsdict = {}


def transform_gen(points, wl):
    """Transform generator function

    Args:
        points (np array): Array of points on the output image (col, row)
        fitpoly (np array): Polynomial coefficients
        col_adj (float): Column adjustment value
        deg (int): Degree of the fit polynomial.

    Returns:
        [type]: [description]
    """
    global pcoeffdict, coladjdict, transformptsdict
    if int(wl*10) not in pcoeffdict.keys():
        raise RuntimeError('poly coefficients do not exist')
    if int(wl*10) not in transformptsdict.keys():
        fitpoly = pcoeffdict[int(wl*10)]
        col_adj = coladjdict[int(wl*10)]
        for i in range(points.shape[0]):
            coord = points[i]
            x = 0
            for i in range(len(fitpoly)):
                x += fitpoly[i]*coord[1]**(len(fitpoly) - 1 - i)
            coord[0] -= col_adj - x
        transformptsdict[int(wl*10)] = points
    else:
        points = transformptsdict[int(wl*10)]

    return points


def straighten_image(img, wl, deg=2):
    """Straighten HiT&MIS image for given segment

    Args:
        img ([type]): [description]
        wl ([type]): [description]
        deg (int, optional): [description]. Defaults to 8.

    Returns:
        [type]: [description]
    """
    global pcoeffdict
    roi = get_roi(wl)  # get ROI
    cimg = img[roi['ymin']:roi['ymax'],
               roi['xmin']:roi['xmax']]  # cropped image
    if int(wl*10) not in pcoeffdict.keys():
        poi = get_lines(wl)
        proj = poi.copy()
        proj[0, :] = proj[0, :].max()
        col_adj = proj[0, :].max()
        pcoeff = np.polyfit(get_lines(wl)[1], get_lines(wl)[
                            0], deg)  # x for given y
        col_max = 0
        for i in range(len(pcoeff)):
            col_max += pcoeff[i]*roi['ymax']**(deg - i)
        pcoeffdict[int(wl*10)] = pcoeff
        projdict[int(wl*10)] = proj
        coladjdict[int(wl*10)] = col_adj
    else:
        proj = projdict[int(wl*10)]
    wimg = transform.warp(cimg, transform_gen, map_args={
                          'wl': wl}, mode='constant', cval=0)
    return (wimg, proj)


# %% Example: random image
if testing:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    imgs = []
    for w in wls:
        img_ = straighten_image(img, w)[0]
        imgs.append(img_)

    hfont = {'fontname': 'Arial'}
    plt.rcParams['font.size'] = 8
    num_filters = len(wls)
    fig = plt.figure(figsize=(720/80, 720/80))
    gs = fig.add_gridspec(2, 3, left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.2, hspace=0.2)
    # fig, ax = plt.subplots(2, 3, squeeze=True, figsize = (720/80, 720/80))
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[0, 1]))
    ax.append(fig.add_subplot(gs[0, 2]))
    ax.append(fig.add_subplot(gs[1, 0:2]))
    ax.append(fig.add_subplot(gs[1, 2]))
    fig.suptitle(" \n ")
    ims = []
    for i in range(num_filters):
        ax[i].title.set_text('%.1f nm filter' % (wls[i]))
        div = make_axes_locatable(ax[i])
        cax = div.append_axes('right', size='5%', pad=0.05)
        im = ax[i].imshow(imgs[i], origin='upper', animated=True)
        fig.colorbar(im, cax=cax)
        ims.append(im)
    plt.show()

# %% Convert time delta to HH:MM:SS string


def tdelta_to_hms(tdelta):
    if tdelta < 0:
        return 'Negative time delta invalid'
    tdelta = int(tdelta)
    tdelta_h = tdelta // 3600
    tdelta -= tdelta_h * 3600
    tdelta_m = tdelta // 60
    tdelta -= tdelta_m * 60
    outstr = ''
    if tdelta_h > 0:
        outstr += str(tdelta_h) + ' h '
    if tdelta_m > 0:
        outstr += str(tdelta_m) + ' m '
    outstr += str(tdelta) + ' s'
    return outstr
# %% Save straightened images, no other processing
def get_imgs_from_files_single(idx, flist: list, wl: float, dark_bias: np.ndarray, dark_rate: np.ndarray, read_noise: float):
    fname = flist[idx]
    try:
        _fimg = pf.open(fname)
    except Exception as e:
        print('Exception %s on file %s' % (str(e), fname))
        return None
    fimg = _fimg[1]
    try:
        data = np.asarray(fimg.data, dtype=float)
        exposure = fimg.header['exposure_ms']*0.001

        # data -= dark_data['bias'] + (dark_data['dark'] * exposure)
        # start = perf_counter_ns()
        # data = unp.uarray(data, np.sqrt(data)) # shot noise
        # end = perf_counter_ns()
        # print('data unp: %.3f us'%((end - start)*1e-3))
        
        # start = perf_counter_ns()
        data = data - dark_bias # remove bias
        std = np.sqrt(data + read_noise*read_noise) # standard deviation measure (Rp * t + Rd * t + Rd^2)
        data = data - (dark_rate * exposure)
        # end = perf_counter_ns()
        # print('data sub: %.3f us'%((end - start)*1e-3))

        # start = perf_counter_ns()
        # std = unp.std_devs(data)
        # data = unp.nominal_values(data)
        # end = perf_counter_ns()
        # print('data break: %.3f us'%((end - start)*1e-3))

        data -= np.average(data[950:, 100:])
        # start = perf_counter_ns()
        data = straighten_image(data, wl)[0]
        std = straighten_image(std, wl)[0]
        data[np.where(data < 0)] = 0
        std[np.where(std < 0)] = 0

        if wl == 557.7: # FOV adjust
            data_ = np.zeros((408, data.shape[1]), dtype = float)
            data_[36:,:] += data[:-10, :]
            del data
            data = data_
            stdval_ = np.zeros((408, data.shape[1]), dtype = float)
            stdval_[36:, :] += std[:-10, :]
            del std
            std = stdval_

        elif wl == 486.1: # FOV adjust
            data_ = np.zeros((425, data.shape[1]), dtype = float)
            data_[:, :] += data[4:-2, :]
            del data
            data = data_
            stdval_ = np.zeros((425, data.shape[1]), dtype = float)
            stdval_[:, :] += std[4:-2, :]
            del std
            std = stdval_
        # end = perf_counter_ns()
        # print('data straighten: %.3f us'%((end - start)*1e-3))

        # start = perf_counter_ns()
        # std = np.sqrt(data) # straighten_image(std, wl)[0]
        # end = perf_counter_ns()
        # print('std straighten: %.3f us'%((end - start)*1e-3))

        # start = perf_counter_ns()
        # data = unp.uarray(data, std)
        # end = perf_counter_ns()
        # print('data pack: %.3f us'%((end - start)*1e-3))
    except Exception as e:
        print('Exception %s on file %s' % (str(e), fname))
        _fimg.close()
        return None

    tstamp = fimg.header['timestamp']*0.001
    return (data, std, tstamp, exposure)

# %%
def get_imgs_from_files(flist, wl, max_workers=12):
    num_frames = len(flist)
    imgdata = []
    stdata = []
    expdata = []
    tdata = []

    if get_imgs_from_files.dark_data is None:
        with lzma.open('pixis_dark_bias.xz', 'rb') as dfile:
            get_imgs_from_files.dark_data = pickle.load(dfile)
            get_imgs_from_files.read_noise = np.median(get_imgs_from_files.dark_data['bias_std'])
    dark_bias: np.ndarray = get_imgs_from_files.dark_data['bias'] # unp.uarray(dark_data['bias'], dark_data['bias_std'])
    dark_rate = get_imgs_from_files.dark_data['dark'] # unp.uarray(dark_data['dark'], dark_data['dark_std'])
    read_noise = get_imgs_from_files.read_noise

    fun = partial(get_imgs_from_files_single, flist=flist, wl=wl, dark_bias=dark_bias, dark_rate=dark_rate, read_noise=read_noise)

    res = thread_map(fun, range(num_frames), max_workers=max_workers)

    for item in res:
        if item is not None:
            imgdata.append(item[0])
            stdata.append(item[1])
            tdata.append(item[2])
            expdata.append(item[3])

    del res
    gc.collect()

    return (imgdata, stdata, expdata, tdata)

    if get_imgs_from_files.dark_data is None:
        with lzma.open('pixis_dark_bias.xz', 'rb') as dfile:
            get_imgs_from_files.dark_data = pickle.load(dfile)
            get_imgs_from_files.read_noise = np.median(get_imgs_from_files.dark_data['bias_std'])
    dark_bias: np.ndarray = get_imgs_from_files.dark_data['bias'] # unp.uarray(dark_data['bias'], dark_data['bias_std'])
    dark_rate = get_imgs_from_files.dark_data['dark'] # unp.uarray(dark_data['dark'], dark_data['dark_std'])
    read_noise = get_imgs_from_files.read_noise

    for i in tqdm(range(num_frames)):
        fname = flist[i]
        try:
            _fimg = pf.open(fname)
        except Exception as e:
            print('Exception %s on file %s' % (str(e), fname))
            continue
        fimg = _fimg[1]
        try:
            data = np.asarray(fimg.data, dtype=float)
            exposure = fimg.header['exposure_ms']*0.001

            # data -= dark_data['bias'] + (dark_data['dark'] * exposure)
            # start = perf_counter_ns()
            # data = unp.uarray(data, np.sqrt(data)) # shot noise
            # end = perf_counter_ns()
            # print('data unp: %.3f us'%((end - start)*1e-3))
            
            # start = perf_counter_ns()
            data = data - dark_bias # remove bias
            std = np.sqrt(data + read_noise*read_noise) # standard deviation measure (Rp * t + Rd * t + Rd^2)
            data = data - (dark_rate * exposure)
            # end = perf_counter_ns()
            # print('data sub: %.3f us'%((end - start)*1e-3))

            # start = perf_counter_ns()
            # std = unp.std_devs(data)
            # data = unp.nominal_values(data)
            # end = perf_counter_ns()
            # print('data break: %.3f us'%((end - start)*1e-3))

            data -= np.average(data[950:, 100:])
            # start = perf_counter_ns()
            data = straighten_image(data, wl)[0]
            std = straighten_image(std, wl)[0]
            data[np.where(data < 0)] = 0
            std[np.where(std < 0)] = 0

            if wl == 557.7: # FOV adjust
                data_ = np.zeros((408, data.shape[1]), dtype = float)
                data_[36:,:] += data[:-10, :]
                data = data_
                stdval_ = np.zeros((408, data.shape[1]), dtype = float)
                stdval_[36:, :] += std[:-10, :]
                std = stdval_

            elif wl == 486.1: # FOV adjust
                data_ = np.zeros((425, data.shape[1]), dtype = float)
                data_[:, :] += data[4:-2, :]
                data = data_
                stdval_ = np.zeros((425, data.shape[1]), dtype = float)
                stdval_[:, :] += std[4:-2, :]
                std = stdval_
            # end = perf_counter_ns()
            # print('data straighten: %.3f us'%((end - start)*1e-3))

            # start = perf_counter_ns()
            # std = np.sqrt(data) # straighten_image(std, wl)[0]
            # end = perf_counter_ns()
            # print('std straighten: %.3f us'%((end - start)*1e-3))

            # start = perf_counter_ns()
            # data = unp.uarray(data, std)
            # end = perf_counter_ns()
            # print('data pack: %.3f us'%((end - start)*1e-3))
        except Exception as e:
            print('Exception %s on file %s' % (str(e), fname))
            continue

        tstamp = (fimg.header['timestamp']*0.001)
        imgdata.append(data)
        stdata.append(std)
        tdata.append(tstamp)
        expdata.append(exposure)

    return (imgdata, stdata, expdata, tdata)

get_imgs_from_files.dark_data = None
get_imgs_from_files.read_noise = None

# %%
# encoding = {'imgs': {'dtype': float, 'zlib': True},
#             'exposure': {'dtype': float, 'zlib': True}}
# imgdata, expdata, tdata = get_imgs_from_files(all_files, wls[0])
# ds = xr.Dataset(
#             data_vars=dict(
#                 imgs=(['tstamp', 'height', 'wl'], imgdata),
#                 exposure=(['tstamp'], expdata)
#             ),
#             coords=dict(tstamp=tdata),
#             attrs=dict(wl=wls[0])
#         )
# fname = 'hitmis_night_%04d.nc'%(wls[0] * 10)
# print('Saving %s...\t' % (fname), end='')
# sys.stdout.flush()
# ds.to_netcdf(destdir + '/' + fname, encoding=encoding)
# print('Done.')
# sys.exit(0)

# %% Save NC files
encoding = {'imgs': {'dtype': float, 'zlib': True},
            'stds': {'dtype': float, 'zlib': True},
            'exposure': {'dtype': float, 'zlib': True}}
for key in main_flist.keys():
    filelist = main_flist[key]
    print('[%04d-%02d-%02d]' % (key.year, key.month, key.day))
    for w in wls:
        fname = 'hitmis_resamp_%04d%02d%02d_%4d.nc' % (
            key.year, key.month, key.day, w * 10)
        if fname in os.listdir(destdir):
            print('File %s exists' % (fname))
            continue
        tstart = datetime.datetime.now().timestamp()
        imgdata, stdata, expdata, tdata = get_imgs_from_files(filelist, w)
        tdelta = datetime.datetime.now().timestamp() - tstart
        try:
            print(' ' * os.get_terminal_size()[0], end='')
        except OSError:
            print(' ' * 80, end = '')
        print('[%.1f] Conversion time: %s' % (w, tdelta_to_hms(tdelta)))
        wl_ax = np.arange(imgdata[0].shape[-1]) * resolutions[w][0] + resolutions[w][1]
        ds = xr.Dataset(
            data_vars=dict(
                imgs=(['tstamp', 'height', 'wl'], imgdata),
                stds=(['tstamp', 'height', 'wl'], stdata),
                exposure=(['tstamp'], expdata)
            ),
            coords=dict(tstamp=tdata, wl=wl_ax),
            attrs=dict(wl=w)
        )
        # do resamp
        start = datetime.datetime.fromtimestamp(tdata[0])
        end = start + datetime.timedelta(0, 240)
        del imgdata
        del stdata
        del expdata
        del tdata
        gc.collect()
        ts = []
        imgs = []
        stdvals = []
        count = 0
        str_len = 0
        while end.timestamp() < ds['tstamp'][-1]:
            # resample
            slc = slice(start.timestamp(), end.timestamp())
            exps = ds.loc[dict(tstamp=slc)]['exposure']
            if exps is not None and len(exps) != 0:
                data: np.ndarray = ds.loc[dict(tstamp=slc)]['imgs']
                std: np.ndarray = ds.loc[dict(tstamp=slc)]['stds']
                # std = np.sqrt(data)
                data = data.sum(axis=0) # all counts
                std = np.sqrt((std**2).sum(axis=0))
                texp = exps.sum() # total exposure
                std = std / texp
                data = data / texp
                # stdev = np.std(data, axis = 0)
                # data = np.average(data, axis = 0)
                # data = data.mean(axis=0)
                # save
                start += datetime.timedelta(0, 120)
                ts.append(start.timestamp())
                imgs.append(data)
                stdvals.append(std)
                count += 1
                p_str = '%s %d'%(str(end), count)
                if str_len:
                    print(' '*str_len, end = '\r')
                print('%s'%(p_str), end = '\r')
                str_len = len(p_str)
            else:
                p_str = '%s: No files'%(str(end))
                if str_len:
                    print(' '*str_len, end = '\r')
                print('%s'%(p_str), end = '\r')
                str_len = len(p_str)
                start += datetime.timedelta(0, 120)
            #update
            start += datetime.timedelta(0, 120)
            end += datetime.timedelta(0, 240)
        del ds
        gc.collect()
        imgs = np.asarray(imgs)
        stdvals = np.asarray(stdvals)
        print('Final:', len(imgs), len(ts))
        encoding = {'imgs': {'dtype': float, 'zlib': True}, 'stds': {'dtype': float, 'zlib': True}}
        nds = xr.Dataset(
                data_vars=dict(
                    imgs=(['tstamp', 'height', 'wl'], imgs),
                    stds=(['tstamp', 'height', 'wl'], stdvals),
                ),
                coords=dict(tstamp=ts, wl=wl_ax)
        )
        print('Saving %s...\t' % (fname), end='')
        sys.stdout.flush()
        nds.to_netcdf(destdir + '/' + fname, encoding=encoding)
        del nds
        del ts
        del wl_ax
        del imgs
        del stdvals
        gc.collect()
        print('Done.')
        print('-' * os.get_terminal_size().columns)
        print('')
    print('')
    print('+' * os.get_terminal_size().columns)
    print('')

# %%
