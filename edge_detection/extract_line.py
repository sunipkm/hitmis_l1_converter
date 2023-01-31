# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# %%
def get_roi(wl, cond = False):
    roi = np.loadtxt('hitmis_roi.csv', skiprows = 1, delimiter = ',').transpose()
    coord = np.where(roi == wl)
    coords = roi[2:, coord[1]]
    try:
        xmin = int(coords[0]) * (2 if cond else 1)
        xmax = int(coords[0] + coords[2]) * (2 if cond else 1)
        ymin = int(coords[1]) * (2 if cond else 1)
        ymax = int(coords[1] + coords[3]) * (2 if cond else 1)
    except Exception:
        return {'xmin': -1, 'ymin': -1, 'xmax': -1, 'ymax': -1}
    return {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}

def get_region(img, wl):
    roi = get_roi(wl)
    return img[roi['ymin']:roi['ymax'], roi['xmin']:roi['xmax']]
# %%
wls = [4278, 4861, 5577, 6300, 6563]
# %%
def get_y(x, poly):
    deg = np.shape(poly)[0] - 1
    res = 0
    for i in range(deg + 1):
        res += poly[i] * x**(deg - i)
    return res

# %%
for wl in wls:
    im = np.asarray(Image.open('%d_edge.png'%(wl)), dtype = float)
    im -= im.min()
    im /= im.max()
    im[np.where(im > 0.2)] = 1
    yx = np.argwhere(im == 1).transpose()
    # fit line
    p = np.polyfit(yx[1], yx[0], deg = 8)
    plt.plot(yx[1], yx[0])
    x = np.linspace(yx[1].min(), yx[1].max(), 100)
    plt.plot(x, get_y(x, p))
    plt.show()
# %%
for wl in wls:
    im = np.asarray(Image.open('%d_edge.png'%(wl)), dtype = float)
    im -= im.min()
    im /= im.max()
    im[np.where(im > 0.2)] = 1
    yx = np.argwhere(im == 1).transpose()
    # fit line
    yx = yx[::-1, :]
    if (yx.shape[0] > 2):
        yx = yx[1:]
    yx = yx.transpose()
    yx = np.asarray(yx, dtype = int)
    print(yx)
    np.savetxt('%d_edge.txt'%(wl), yx, fmt = '%i')
# %%
