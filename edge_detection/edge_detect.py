# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
# %%
# 1. Load ROI
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
rawimg = cv2.imread('hitmis_daycalib_2.png')
plt.imshow(get_region(rawimg, 486.1))
# %%
sobelxy = cv2.Sobel(src=get_region(rawimg, 486.1), ddepth=cv2.CV_64F, dx=1, dy=1, ksize=11)
plt.imshow(sobelxy)
cv2.imwrite('4861_edge.png', sobelxy)
# %%
edges = cv2.Canny(image=get_region(rawimg, 486.1), threshold1=20, threshold2=180)
plt.imshow(edges)
# %%
edges = cv2.Canny(image=get_region(rawimg, 656.3), threshold1=50, threshold2=120)
plt.imshow(edges)
# %%
sobelxy = cv2.Sobel(src=get_region(rawimg, 656.3), ddepth=cv2.CV_64F, dx=1, dy=1, ksize=11)
plt.imshow(sobelxy)
cv2.imwrite('6563_edge.png', sobelxy)
# %%
edges = cv2.Canny(image=get_region(rawimg, 427.8), threshold1=40, threshold2=120)
plt.imshow(edges)
# %%
sobelxy = cv2.Sobel(src=get_region(rawimg, 427.8), ddepth=cv2.CV_64F, dx=1, dy=1, ksize=11)
plt.imshow(sobelxy)
cv2.imwrite('4278_edge.png', sobelxy) 
# %%
edges = cv2.Canny(image=get_region(rawimg, 557.7), threshold1=80, threshold2=100) 
plt.imshow(edges)
# %%
sobelxy = cv2.Sobel(src=get_region(rawimg, 557.7), ddepth=cv2.CV_64F, dx=1, dy=1, ksize=11)
cv2.imwrite('5577_edge.png', sobelxy)
plt.imshow(sobelxy)
# %%
sobelxy = cv2.Sobel(src=get_region(rawimg, 630), ddepth=cv2.CV_64F, dx=1, dy=1, ksize=11)
plt.imshow(sobelxy)
cv2.imwrite('6300_edge.png', sobelxy)
# %%
from PIL import Image
rawimg = np.asarray(Image.open('hitmis_calib.png'),dtype = float)
plt.imshow(rawimg)
rawimg -= rawimg.min()
rawimg *= 1.0 / np.max(rawimg)
rawimg *= 255
rawimg = np.uint8(rawimg)
# %%
edges = cv2.Canny(image=get_region(rawimg, 557.7), threshold1=240, threshold2=250) 
cv2.imwrite('5577_edge.png', edges)
plt.imshow(edges)
# %%
edges = cv2.Canny(image=get_region(rawimg, 630), threshold1=248, threshold2=248) 
cv2.imwrite('6300_edge.png', edges)
plt.imshow(edges)
# %%
sobelxy = cv2.Sobel(src=get_region(rawimg, 630), ddepth=cv2.CV_64F, dx=1, dy=1, ksize=25)
plt.imshow(sobelxy)
# %%
sobelxy -= sobelxy.min()
sobelxy /= sobelxy.max()
sobelxy *= 255
# %%
plt.imshow(sobelxy)
plt.colorbar()
plt.show()
# %%
edges = cv2.Canny(image=np.uint8(sobelxy), threshold1=120, threshold2=130) 
# cv2.imwrite('6300_edge.png', edges)
plt.imshow(edges)
# %%
