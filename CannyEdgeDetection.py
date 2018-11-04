# Canny Edge Detection - Eren ATAS - 1334129
# Python 3.6 is used.

# Import Libs
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
from matplotlib import pyplot as plt
import numpy as np
import imageio
import optparse

# Set figure settings
plt.rcParams["figure.figsize"] = (15, 20)


# Set optParse
parser = optparse.OptionParser()

parser.add_option('--image',
                  action="store", dest="imgName",
                  help="Path of the image file", default="im01.jpg")

parser.add_option('--sigma', type='float',
                  action="store", dest="sigmaArg",
                  help="standard deviation of sigma", default=1.4)


parser.add_option('--minthreshold', type='int',
                  action="store", dest="minTArg",
                  help="Minimum Threshold for Double Thresholding", default=100)

parser.add_option('--maxthreshold', type='int',
                  action="store", dest="maxTArg",
                  help="Maximum Threshold for Double Thresholding", default=200)

options, args = parser.parse_args()


# Get Arguments
imageName = options.imgName
sigma = options.sigmaArg
minT = options.minTArg
maxT = options.maxTArg
gaussianWindow = 5
high = 255
low = 50

# Step 1: Smoothing
img = imageio.imread(imageName, as_gray=True)
img = img.astype('int32')
t = (((gaussianWindow - 1) / 2) - 0.5) / sigma
img1 = gaussian_filter(img, sigma, truncate=t)
plt.subplot(2, 2, 1), plt.imshow(img1, cmap='gray')
plt.title('Gaussian filter')

sobelX = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]], np.int32
)
sobelX.shape

# Step 2: Finding gradients
sobelX = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]], np.int32
)

sobelY = np.array(
    [[1, 2, 1],
     [0, 0, 0],
     [-1, -2, -1]], np.int32
)

dX = ndimage.filters.convolve(img1, sobelX)
dY = ndimage.filters.convolve(img1, sobelY)

# Finding gradient and theta
img2 = np.hypot(dX, dY)
d = np.arctan2(dY, dX)

# Step 3: Non-maximum suppression
m, n = img2.shape
z = np.zeros((m, n), dtype=np.int32)

for i in range(m):
    for j in range(n):
        # find neighbour pixels to visit from the gradient directions
        where = np.rad2deg(d[i, j]) % 180
        if (0 <= d[i, j] < 22.5) or (157.5 <= d[i, j] < 180):
            d[i, j] = 0
        elif (22.5 <= d[i, j] < 67.5):
            d[i, j] = 45
        elif (67.5 <= d[i, j] < 112.5):
            d[i, j] = 90
        elif (112.5 <= d[i, j] < 157.5):
            d[i, j] = 135

        try:
            if where == 0:
                if (img2[i, j] >= img2[i, j - 1]) and (img2[i, j] >= img2[i, j + 1]):
                    z[i, j] = img2[i, j]
            elif where == 45:
                if (img2[i, j] >= img2[i - 1, j + 1]) and (img2[i, j] >= img2[i + 1, j - 1]):
                    z[i, j] = img2[i, j]
            elif where == 90:
                if (img2[i, j] >= img2[i - 1, j]) and (img2[i, j] >= img2[i + 1, j]):
                    z[i, j] = img2[i, j]
            elif where == 135:
                if (img2[i, j] >= img2[i - 1, j - 1]) and (img2[i, j] >= img2[i + 1, j + 1]):
                    z[i, j] = img2[i, j]
        except IndexError as e:
            'find out why it\'s giving exception'
            pass

plt.subplot(2, 2, 2), plt.imshow(img2, cmap='gray')
plt.title('Non-maximum suppression')

# Step 4: Double thresholding
img3 = img2
highX, highY = np.where(img3 > maxT)
lowX, lowY = np.where((img3 >= minT) & (img3 <= maxT))
zeroX, zeroY = np.where(img3 < minT)

img3[highX, highY] = high
img3[lowX, lowY] = low
img3[zeroX, zeroY] = np.int32(0)

plt.subplot(2, 2, 3), plt.imshow(img3, cmap='gray')
plt.title('Double Thresholding')


# Step 5: Edge tracking by hysteresis
img4 = img3
m, n = img4.shape

for i in range(m):
    for j in range(n):
        if img[i, j] == low:
            try:
                if ((img4[i + 1, j] == high) or (img4[i - 1, j] == high)
                        or (img4[i, j + 1] == high) or (img4[i, j - 1] == high)
                        or (img4[i + 1, j + 1] == high) or (img4[i - 1, j - 1] == high)):
                    img[i, j] = high
                else:
                    img[i, j] = 0
            except IndexError as e:
                pass

plt.subplot(2, 2, 4), plt.imshow(img4, cmap='gray')
plt.title('Edge Tracking by Histeresis')


plt.show()
