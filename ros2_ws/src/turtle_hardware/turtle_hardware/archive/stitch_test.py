# import the necessary packages
# from __future__ import print_function
# from basicmotiondetector import BasicMotionDetector
from stitching import Stitcher
# from imutils.video import VideoStream
import numpy as np
import datetime
# import imutils
import time
import cv2

# initialize the video streams and allow them to warmup
# print("[INFO] starting cameras...")
# leftStream = cv2.VideoCapture("a2.mp4")
# rightStream = cv2.VideoCapture("a1.mp4")
right = cv2.imread("right/right 1.png")
left = cv2.imread("left/left 1.png")

time.sleep(2.0)

# initialize the image stitcher, motion detector, and total
# number of frames read
stitcher = Stitcher()
# motion = BasicMotionDetector(minArea=500)
# total = 0

# loop over frames from the video streams
# while True:
# grab the frames from their respective video streams
# ret, left = leftStream.read()
# ret1, right = rightStream.read()



result = stitcher.stitch([left, right])
# no homograpy could be computed
if result is None:
    print("[INFO] homography could not be computed")
# convert the panorama to grayscale, blur it slightly, update
# the motion detector
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (21, 21), 0)

# increment the total number of frames read and draw the 
# timestamp on the image

# show the output images
cv2.imshow("Result", result)
cv2.imshow("Left Frame", left)
cv2.imshow("Right Frame", right)
cv2.waitKey()
# if the `q` key was pressed, break from the loop
# if key == ord("q"):
#     break
# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
# leftStream.stop()
# rightStream.stop()