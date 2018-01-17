import skimage as ski
import skimage.io 
import skimage.filters
import skimage.color
import skimage.util
import numpy as np
import argparse
import os.path as path


parser = argparse.ArgumentParser(description='Apply desired filter to image and convert to light mask')
parser.add_argument('imgpath')
parser.add_argument('maskpath', nargs='?', default=None)
parser.add_argument('--invert', '-i', nargs=1, default = False)
parser.add_argument('--binary', '-b', nargs=1, type=bool, default=False)
parser.add_argument('--blur', '-u', nargs=1, type=float, default=0)
#Sharpening TBD
#parser.add_argument('--sharpen', nargs='?', type=float, default=0)
args = parser.parse_args()
imgpath = args.imgpath
maskpath = args.maskpath
binary = args.binary
blur = args.blur
invert = args.invert


# load image
img = ski.io.imread(imgpath)

# Convert image to grayscale
img = ski.color.rgb2gray(img)

# Invert image
if invert:
    img = ski.util.invert(img)

# Binarize if option

if binary:
    img = ski.filters.threshold_otsu(img)

# Blur/Sharpen if necessary

if blur > 0 :
    img = ski.filters.gaussian(img, sigma=blur)

# Sharpening TBD
#if sharpen > 0:
#    img = img - 0.3*ski.filters.gaussian(img, sigma=sharpen)

# Convert image to 8-bit
img = ski.img_as_ubyte(img)

# Save image to output path
(input_filename, ext) = path.splitext(path.basename(imgpath))
output_filename = 'mask_' + input_filename + '.tif'

if maskpath is None:
    outputdir = path.dirname(imgpath)
else:
    outputdir = maskpath

outputpath = path.join(outputdir, output_filename)
ski.io.imsave(outputpath, img)

