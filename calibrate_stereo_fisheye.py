import cv2
assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob

CHECKERBOARD = (6,9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
count=54

objp = np.zeros( (CHECKERBOARD[0]*CHECKERBOARD[1], 1, 3) , np.float64)
objp[:,0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] # 3d point in real world space
leftimgpoints = [] # 2d points in image plane.
rightimgpoints = [] # 2d points in image plane.
leftimages = sorted(glob.glob('L*.jpeg'))
rightimages = sorted(glob.glob('R*.jpeg'))
for i in range(len(leftimages)):
    leftname=leftimages[i]
    rightname=rightimages[i]

    leftimg = cv2.imread(leftname)
    rightimg=cv2.imread(rightname)

    leftgray = cv2.cvtColor(leftimg,cv2.COLOR_BGR2GRAY)
    rightgray = cv2.cvtColor(rightimg,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    lret, lcorners = cv2.findChessboardCorners(leftgray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    rret, rcorners = cv2.findChessboardCorners(rightgray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if lret == True and rret==True:
        objpoints.append(objp)
        cv2.cornerSubPix(leftgray,lcorners,(3,3),(-1,-1),subpix_criteria)
        leftimgpoints.append(lcorners)
        cv2.cornerSubPix(rightgray,rcorners,(3,3),(-1,-1),subpix_criteria)
        rightimgpoints.append(rcorners)

    print(leftgray.shape[::-1])

# LR_num=len(leftimgpoints)
# #objpoints=np.float32([(np.reshape(objpoints,(54*len(leftimgpoints),3)))])
objpoints = np.array(objpoints, np.float64)
leftimgpoints = np.array(leftimgpoints, np.float64)
rightimgpoints = np.array(rightimgpoints, np.float64)
# leftimgpoints=np.float64(np.reshape(leftimgpoints,(count,LR_num,2)))
# rightimgpoints=np.float64(np.reshape(rightimgpoints,(count,LR_num,2)))
print(objpoints.shape,leftimgpoints.shape,rightimgpoints.shape)

N_OK = len(objpoints)
KL=np.array([[936.3090354816636, 0.0, 1011.6603031360017], [0.0, 936.2302136924422, 542.9121240301195], [0.0, 0.0, 1.0]])
DL=np.array([[-0.0697808462865659], [-0.0031580486418187653], [0.0024539254744271204], [-0.0012931220378302362]])
KR=np.array([[935.5251123185054, 0.0, 990.8427164459354], [0.0, 936.1743386189506, 536.9829760404886], [0.0, 0.0, 1.0]])
DR=np.array([[-0.08446254512511926], [0.036252030276062254], [-0.03991208556882756], [0.01427235965589032]])
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, KL, DL, KR, DR, R, T,rvecs,tvecs = cv2.fisheye.stereoCalibrate(
        objpoints,
        leftimgpoints,
        rightimgpoints,
        KL,
        DL,
        KR,DR,
        leftgray.shape[::-1]
    )

print("Found " + str(N_OK) + " valid images for calibration")
print("KL=np.array(" + str(KL.tolist()) + ")")
print("DL=np.array(" + str(DL.tolist()) + ")")
print("KR=np.array(" + str(KR.tolist()) + ")")
print("DR=np.array(" + str(DR.tolist()) + ")")
print("R=np.array(" + str(R.tolist()) + ")")
print("T=np.array(" + str(T.tolist()) + ")")