
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
imgL = cv.imread('0LZed_0.png', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('0RZed_0.png', cv.IMREAD_GRAYSCALE)
# window_size = 3
# n=1
# m=5
# min_disp = 16*n
# num_disp = 16*m
stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
# stereo.setMinDisparity(min_disp)
# # stereo.setPreFilterSize(255)
# #^must be odd & btwn 5 & 255
# stereo.setPreFilterCap(20)
# #^must be btwn 0 and 63
# stereo.setSpeckleWindowSize(32)
# # stereo.setTextureThreshold(100)
# stereo.setUniquenessRatio(3)
# stereo.setSpeckleRange(50)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity/255,'gray')
plt.show()

