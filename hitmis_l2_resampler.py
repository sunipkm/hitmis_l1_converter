# %% Imports
import os
import sys
import glob
import astropy.io.fits as pf
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from skimage import transform
import datetime
import argparse
import lzma
import pickle
# %% Argument Parser
parser = argparse.ArgumentParser(
    description='Convert HiT&MIS L0 data to L2 data, with exposure normalization and timestamp regularization (1 image for every 4 minutes), which separates data by filter region and performs line straightening using ROI listed in accompanying "hitmis_roi.csv" file, and edge lines in "edge_detection" directory. The program will not work without these files present.')
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
wls = [630.0, 557.7]
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


def get_imgs_from_files(flist):
    num_frames = len(flist)
    otstamp = None
    tconsume = 0
    imgs = {}
    stds = {}
    expdata = []
    tdata = []

    for wl in wls:
        locals()['imgdata_%d'%(int(wl * 10))] = []
        locals()['stdata_%d'%(int(wl * 10))] = []

    with lzma.open('pixis_dark_bias.xz', 'rb') as dfile:
        dark_data = pickle.load(dfile)

    for i in range(num_frames):
        if otstamp is None:
            otstamp = datetime.datetime.timestamp(datetime.datetime.now())
        if (i + 1) % 20 == 0:
            ctstamp = datetime.datetime.timestamp(datetime.datetime.now())
            tdelta = ctstamp - otstamp
            otstamp = ctstamp
            tconsume += tdelta
            ttot = ((num_frames - i - 2) * tdelta / 20) + tconsume
            str_p1 = "Processing... [%d/%d]" % (i + 1, num_frames)
            str_p2 = "[%s/%s]" % (tdelta_to_hms(tconsume), tdelta_to_hms(ttot))
            width = os.get_terminal_size().columns
            n_spaces = width - len(str_p1) - len(str_p2)
            if n_spaces <= 4:
                print("%s\t%s" % (str_p1, str_p2))
            else:
                spaces = ' ' * n_spaces
                print('%s%s%s' % (str_p1, spaces, str_p2), end='\r')
        fname = flist[i]
        try:
            _fimg = pf.open(fname)
        except Exception as e:
            print('Exception %s on file %s' % (str(e), fname))
            continue
        fimg = _fimg[1]
        try:
            rdata = np.asarray(fimg.data, dtype=float)
            tstamp = (fimg.header['timestamp']*0.001)
            exposure = fimg.header['exposure_ms']*0.001
        except Exception as e:
            print('Exception %s on file %s' % (str(e), fname))
            continue

        rstdval = np.sqrt(rdata)
        rdata -= dark_data['bias'] + (dark_data['dark'] * exposure)
        rdata -= np.average(rdata[950:, 100:])

        for wl in wls:
            data = straighten_image(rdata, wl)[0]
            stdval = straighten_image(rstdval, wl)[0]

            if wl == 557.7: # FOV adjust
                data_ = np.zeros((408, data.shape[1]), dtype = float)
                data_[36:,:] += data[:-10, :]
                data = data_
                stdval_ = np.zeros((408, data.shape[1]), dtype = float)
                stdval_[36:, :] += stdval[:-10, :]
                stdval = stdval_

            data = skimage.measure.block_reduce(data, (10, 1), np.sum) # np.mean)
            stdval = skimage.measure.block_reduce(stdval, (10, 1), np.sum) # np.mean)
        
            locals()['imgdata_%d'%(int(wl * 10))].append(data)
            locals()['stdata_%d'%(int(wl * 10))].append(stdval)
        
        for wl in wls:
            imgs[wl] = locals()['imgdata_%d'%(int(wl * 10))]
            stds[wl] = locals()['stdata_%d'%(int(wl * 10))]

        tdata.append(tstamp)
        expdata.append(exposure)

    return (imgs, stds, expdata, tdata)

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
import skimage.measure

encoding = {'imgs': {'dtype': float, 'zlib': True},
            'exposure': {'dtype': float, 'zlib': True}}
for key in main_flist.keys():
    filelist = main_flist[key]
    print('[%04d-%02d-%02d] Starting conversion...' % (key.year, key.month, key.day))
    print(' ' * os.get_terminal_size()[0], end='')

    tstart = datetime.datetime.now().timestamp()
    wimgs, wstds, expdata, tdata = get_imgs_from_files(filelist)

    tdelta = datetime.datetime.now().timestamp() - tstart
    print(' ' * os.get_terminal_size()[0], end='')
    print('Conversion time: %s' % (tdelta_to_hms(tdelta)))

    for w in wls:
        fname = 'hitmis_dsamp_%04d%02d%02d_%4d.nc' % (
            key.year, key.month, key.day, w * 10)
        if fname in os.listdir(destdir):
            print('File %s exists' % (fname))
            continue
        imgdata = wimgs[w]
        stdata = wstds[w]

        x_ax = np.linspace(74, 27, imgdata[0].shape[0], endpoint=False)
        wl_ax = np.arange(imgdata[0].shape[-1]) * resolutions[w][0] + resolutions[w][1]
        ds = xr.Dataset(
            data_vars=dict(
                imgs=(['tstamp', 'height', 'wl'], imgdata),
                stds=(['tstamp', 'height', 'wl'], stdata),
                exposure=(['tstamp'], expdata)
            ),
            coords=dict(tstamp=tdata, wl=wl_ax, height=x_ax),
            attrs=dict(wl=w)
        )
        # do resamp
        start = datetime.datetime.fromtimestamp(tdata[0])
        end = start + datetime.timedelta(0, 240)
        del imgdata
        del stdata
        ts = []
        imgs = []
        stds = []
        count = 0
        str_len = 0
        while end.timestamp() < ds['tstamp'][-1]:
            # resample
            slc = slice(start.timestamp(), end.timestamp())
            exps = ds.loc[dict(tstamp=slc)]['exposure']
            if exps is not None and len(exps) != 0:
                data = ds.loc[dict(tstamp=slc)]['imgs']
                stdev = ds.loc[dict(tstamp=slc)]['stds']
                data /= exps
                stdev /= exps
                data = np.average(data, axis = 0)
                stdev = np.average(stdev, axis = 0)
                # save
                start += datetime.timedelta(0, 120)
                ts.append(start.timestamp())
                imgs.append(data)
                stds.append(data)
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
        print('Final:', len(imgs), len(ts))
        encoding = {'imgs': {'dtype': float, 'zlib': True}}
        nds = xr.Dataset(
                data_vars=dict(
                    imgs=(['tstamp', 'height', 'wl'], imgs),
                    stds=(['tstamp', 'height', 'wl'], stds)
                ),
                coords=dict(tstamp=ts, wl=wl_ax, height=x_ax)
        )
        print('Saving %s...\t' % (fname), end='')
        sys.stdout.flush()
        nds.to_netcdf(destdir + '/' + fname, encoding=encoding)
        print('[%.2f MiB] Done.'%(os.path.getsize(destdir + '/' + fname) / 1024 / 1024))

# %%
