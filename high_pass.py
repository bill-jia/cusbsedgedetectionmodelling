import skimage as ski
import skimage.io 
import skimage.filters
import skimage.color
import skimage.util
import skimage.transform
import numpy as np
import argparse
import os.path as path

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Apply desired filter to image and convert to light mask')
parser.add_argument('imgpath')
parser.add_argument('--alpha', '-a', nargs=1, type=float, default=1)
parser.add_argument('--beta', '-b', nargs=1, type=float, default=1)
parser.add_argument('--sigma', '-s', nargs=1, type=float, default=1)
parser.add_argument('--maxlight', '-m', nargs=1, type=float, default=0.5)

args = parser.parse_args()
imgpath = args.imgpath
alpha = args.alpha
beta = args.beta
sigma = args.sigma
max_light = args.maxlight

# load image
img = ski.io.imread(imgpath)
img = img*max_light/255
img = ski.transform.rotate(img, 180)

img_filt = alpha*img - beta*ski.filters.gaussian(img, sigma)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
colors = plt.pcolormesh(img_filt)
plt.colorbar(colors)
plt.grid()
plt.show()

img_filt = ski.filters.laplace(img)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
colors = plt.pcolormesh(img_filt)
plt.colorbar(colors)
plt.grid()
plt.show()