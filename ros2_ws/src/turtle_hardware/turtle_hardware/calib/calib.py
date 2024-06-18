import cv2
# assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob

CHECKERBOARD = (6,9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
count=63

objp = np.zeros( (CHECKERBOARD[0]*CHECKERBOARD[1], 1, 3) , np.float64)
objp[:,0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] # 3d point in real world space
leftimgpoints = [] # 2d points in image plane.
rightimgpoints = [] # 2d points in image plane.
rightimages = sorted(glob.glob('left *.png'))
leftimages = sorted(glob.glob('right *.png'))
print(rightimages)
# leftimages = ['L_1.png', 'L_2.png', 'L_0.png', 'L_3.png', 'L_4.png', 'L_5.png', 'L_6.png', 'L_7.png', 'L_8.png', 'L_9.png', 'L_11.png', 'L_12.png', 'L_10.png', 'L_13.png', 'L_14.png', 'L_15.png', 'L_16.png', 'L_17.png', 'L_18.png', 'L_19.png', 'L_21.png', 'L_22.png', 'L_20.png', 'L_23.png', 'L_24.png', 'L_25.png', 'L_26.png', 'L_27.png', 'L_28.png', 'L_29.png', 'L_30.png', 'L_33.png', 'L_34.png', 'L_35.png', 'L_36.png', 'L_37.png', 'L_38.png', 'L_39.png', 'L_41.png', 'L_42.png', 'L_40.png', 'L_43.png', 'L_44.png', 'L_45.png', 'L_46.png', 'L_47.png', 'L_48.png', 'L_49.png', 'L_50.png', 'L_51.png', 'L_52.png']
# rightimages = ['R_1.png', 'R_2.png', 'R_0.png', 'R_3.png', 'R_4.png', 'R_5.png', 'R_6.png', 'R_7.png', 'R_8.png', 'R_9.png', 'R_11.png', 'R_12.png', 'R_10.png', 'R_13.png', 'R_14.png', 'R_15.png', 'R_16.png', 'R_17.png', 'R_18.png', 'R_19.png','R_21.png', 'R_22.png', 'R_20.png', 'R_23.png', 'R_24.png', 'R_25.png', 'R_26.png', 'R_27.png', 'R_28.png', 'R_29.png', 'R_30.png', 'R_33.png', 'R_34.png', 'R_35.png', 'R_36.png', 'R_37.png', 'R_38.png', 'R_39.png', 'R_41.png', 'R_42.png', 'R_40.png', 'R_43.png', 'R_44.png', 'R_45.png', 'R_46.png', 'R_47.png', 'R_48.png', 'R_49.png', 'R_50.png', 'R_51.png',  'R_52.png']
for i in range(len(leftimages)):
    leftname=leftimages[i]
    rightname=rightimages[i]

    leftimg = cv2.imread(leftname)
    rightimg=cv2.imread(rightname)

    leftgray = cv2.cvtColor(leftimg,cv2.COLOR_BGR2GRAY)
    rightgray = cv2.cvtColor(rightimg,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('left',leftimg)
    # cv2.waitKey(1000)
    # Find the chess board corners
    lret, lcorners = cv2.findChessboardCorners(leftgray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    rret, rcorners = cv2.findChessboardCorners(rightgray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # cv2.drawChessboardCorners(leftimg, CHECKERBOARD, lcorners, lret)
    # cv2.imshow('img left', leftimg)
    # cv2.drawChessboardCorners(rightimg, CHECKERBOARD, rcorners, rret)
    # cv2.imshow('img right', rightimg)
    # cv2.waitKey()

    # If found, add object points, image points (after refining them)
    if lret == True and rret==True:
        cv2.cornerSubPix(leftgray,lcorners,(3,3),(-1,-1),subpix_criteria)
        cv2.cornerSubPix(rightgray,rcorners,(3,3),(-1,-1),subpix_criteria)
        objpoints.append(objp)
        leftimgpoints.append(lcorners)
        rightimgpoints.append(rcorners)

LR_num=len(leftimgpoints)
objpoints = np.array(objpoints, np.float64)
leftimgpoints = np.array(leftimgpoints, np.float64)
rightimgpoints = np.array(rightimgpoints, np.float64)

# objpoints=np.reshape(objpoints,(LR_num, 1, count,3))
# leftimgpoints=np.reshape(leftimgpoints,(LR_num, 1, count, 2))
# rightimgpoints=np.reshape(rightimgpoints,(LR_num, 1, count, 2))

retL, KL, DL, _, _ = cv2.fisheye.calibrate(objpoints, leftimgpoints, leftgray.shape[::-1], None, None)
retR, KR, DR, _, _ = cv2.fisheye.calibrate(objpoints, rightimgpoints, rightgray.shape[::-1], None, None)


N_OK = len(objpoints)

print("Found " + str(N_OK) + " valid images for calibration")
print("KL=np.array(" + str(KL.tolist()) + ")")
print("DL=np.array(" + str(DL.tolist()) + ")")
print("KR=np.array(" + str(KR.tolist()) + ")")
print("DR=np.array(" + str(DR.tolist()) + ")")
# KL=np.array([[417.15751114066205, 0.0, 336.595336628034], [0.0, 416.8576501537559, 241.5489118345027], [0.0, 0.0, 1.0]])
# DL=np.array([[-0.06815812211170555], [-0.016732544509364528], [0.029182156593969097], [-0.017701284426359723]])
DIM=(640, 480)
# KR=np.array([[416.3903560278583, 0.0, 343.1831889045121], [0.0, 415.88140111385025, 241.99492603370734], [0.0, 0.0, 1.0]])
# DR=np.array([[-0.06197454939758593], [-0.031440749408005376], [0.04248811930174599], [-0.02113466201121944]])
# rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
# tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
# flags = cv.CALIB_FIX_INTRINSIC

rms, KL, DL, KR, DR, R, T = cv2.fisheye.stereoCalibrate(
        objpoints,
        leftimgpoints,
        rightimgpoints,
        KL,
        DL,
        KR,DR,
        leftgray.shape[::-1], criteria=subpix_criteria, flags=calibration_flags
    )

print("Found " + str(N_OK) + " valid images for calibration")
print("KL=np.array(" + str(KL.tolist()) + ")")
print("DL=np.array(" + str(DL.tolist()) + ")")
print("KR=np.array(" + str(KR.tolist()) + ")")
print("DR=np.array(" + str(DR.tolist()) + ")")
print("R=np.array(" + str(R.tolist()) + ")")
print("T=np.array(" + str(T.tolist()) + ")")

R1,R2,P1,P2,Q = cv2.fisheye.stereoRectify(KL,DL,KR,DR,DIM,R,T, cv2.fisheye.CALIB_ZERO_DISPARITY)
print(Q)
L_undist_map=cv2.fisheye.initUndistortRectifyMap(KL,DL,np.identity(3),KL,DIM,cv2.CV_32FC1)
R_undist_map=cv2.fisheye.initUndistortRectifyMap(KR,DR,np.identity(3),KR,DIM,cv2.CV_32FC1)
left1, left2 = cv2.fisheye.initUndistortRectifyMap(KL, DL, R1, P1, DIM, cv2.CV_32FC1)
right1, right2 = cv2.fisheye.initUndistortRectifyMap(KR,DR, R2, P2, DIM, cv2.CV_32FC1)
stereo = cv2.StereoBM.create(numDisparities=128, blockSize=67)
stereo.setMinDisparity(0)
stereo.setTextureThreshold(0)

#post filtering parameters: prevent false matches, help filter at boundaries
stereo.setSpeckleRange(2)
stereo.setSpeckleWindowSize(5)
stereo.setUniquenessRatio(2)

stereo.setDisp12MaxDiff(2)

left = cv2.imread(leftname)
right=cv2.imread(rightname)

fixedLeft = cv2.remap(left, left1, left2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
fixedRight = cv2.remap(right, right1, right2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
# cv.imshow("fixedLeft", fixedLeft)
# cv.imshow("fixedRight", fixedRight)

grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
disparity = stereo.compute(grayLeft,grayRight)
# denoise = 5
# noise=cv2.erode(disparity,np.ones((denoise,denoise)))
# noise=cv2.dilate(noise,np.ones((denoise,denoise)))
# blur = cv2.GaussianBlur(noise, (3,3), 1)
# norm_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
valid_pixels = disparity > 0
invalid_pixels = disparity < 0
disparity[invalid_pixels] = 0.0001
norm_disparity = np.array((disparity/16.0 - stereo.getMinDisparity())/stereo.getNumDisparities(), dtype='f')
points3D = cv2.reprojectImageTo3D(np.array(disparity/16.0, dtype='f'),Q)
depth = Q[2,3]/Q[3,2]/np.array(disparity/16.0, dtype='f')
print(disparity/16.0)
cv2.imshow('img', depth)
cv2.waitKey()