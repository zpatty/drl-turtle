import cv2
# assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


CHECKERBOARD = (6,9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
count=63

objp = np.zeros( (CHECKERBOARD[0]*CHECKERBOARD[1], 1, 3) , np.float64)
objp[:,0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] # 3d point in real world space
leftimgpoints = [] # 2d points in image plane.
rightimgpoints = [] # 2d points in image plane.
rightimages = sorted(glob.glob('right/*.jpg'))
leftimages = sorted(glob.glob('left/*.jpg'))
print('right/ *.jpg')
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

rms, KL, DL, KR, DR, R, T, rvecs, tvecs = cv2.fisheye.stereoCalibrate(
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

T = T*25
R1,R2,P1,P2,Q = cv2.fisheye.stereoRectify(KL,DL,KR,DR,DIM,R,T, cv2.fisheye.CALIB_ZERO_DISPARITY)
print(Q)
L_undist_map=cv2.fisheye.initUndistortRectifyMap(KL,DL,np.identity(3),KL,DIM,cv2.CV_32FC1)
R_undist_map=cv2.fisheye.initUndistortRectifyMap(KR,DR,np.identity(3),KR,DIM,cv2.CV_32FC1)
left1, left2 = cv2.fisheye.initUndistortRectifyMap(KL, DL, R1, P1, DIM, cv2.CV_32FC1)
right1, right2 = cv2.fisheye.initUndistortRectifyMap(KR,DR, R2, P2, DIM, cv2.CV_32FC1)
stereo = cv2.StereoBM.create(numDisparities=128, blockSize=77)
stereo.setMinDisparity(0)
stereo.setTextureThreshold(0)

#post filtering parameters: prevent false matches, help filter at boundaries
stereo.setSpeckleRange(2)
stereo.setSpeckleWindowSize(5)
stereo.setUniquenessRatio(5)

stereo.setDisp12MaxDiff(2)

left = cv2.imread("left/frame1.jpg")
right=cv2.imread("right/frame1.jpg")

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
# disparity = cv2.medianBlur(disparity, ksize=5)
# norm_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
valid_pixels = disparity > 0.0
invalid_pixels = disparity < 0.0001
# print(np.shape(invalid_pixels))
# print(disparity)
disparity[invalid_pixels] = 0
# print(disparity)
# disparity[disparity > 500] = 1
norm_disparity = np.array((disparity/16.0 - stereo.getMinDisparity())/stereo.getNumDisparities(), dtype='f')
points3D = cv2.reprojectImageTo3D(np.array(disparity/16.0/1000, dtype='f'), Q, handleMissingValues=True)
depth = Q[2,3]/Q[3,2]/np.array(disparity/16.0, dtype='f')/1000
depth[np.isinf(depth)] = np.max(depth[np.isfinite(depth)])
# print(Q[2,3])
# print(1/Q[3,2])
# print(np.min(disparity/16.0))
# print(len(np.unique(disparity)))
# finiteX = points3D[np.isfinite(points3D)]
# print(finiteX[:,:,0])
# print(np.min(depth[np.isfinite(depth)]))
# print(points3D[:,:,:])
# print(points3D[200,200,1])
# print(points3D[200,200,2])

# plt.imshow((points3D[:,:,0]**2 + points3D[:,:,1]**2 + points3D[:,:,2]**2)/1000)
# plt.imshow(disparity/16.0)
# print(np.median(depth/1000))
im = plt.imshow(depth)

plt.show()
print(np.shape(points3D))
# opencv loads the image in BGR, so lets make it RGB
colors = np.reshape(cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2RGB), (-1,3))
print(np.shape(colors))


# points3D[:,:,2] = depth
# X = points3D[:,:,0]
# X[np.isinf(X)] = np.max(X[np.isfinite(X)])
# Y = points3D[:,:,1]
# Y[np.isinf(Y)] = np.max(Y[np.isfinite(Y)])
# points3D[:,:,0] = X
# points3D[:,:,1] = Y
# print(np.shape(np.reshape(points3D, (-1,3))))
points3D = np.array([[[10,20,30],[3,5,3.2]]], dtype='f')
test_pts = np.reshape(points3D, (-1,3)).T
projected_points,_ = cv2.projectPoints(np.reshape(points3D, (-1,3)), R, np.array(T/1000, dtype='f'), KL, DL)
projected_points,_ = cv2.projectPoints(np.reshape(points3D, (-1,3)), np.identity(3), np.array([0., 0., 0.]), \
                          KL, np.array([0., 0., 0., 0.]))
print(projected_points)
man_proj = KL @ test_pts
man_proj = np.divide(man_proj,man_proj[-1,:])
print(man_proj)
# print(np.shape(projected_points.reshape(np.shape(points3D)[0], -1, 2)))
# projected_img = projected_points.reshape(np.shape(points3D)[0], -1, 2)

blank_img = np.zeros(fixedLeft.shape, 'uint8')

for i, pt in enumerate(projected_points):
    if np.isfinite(pt).all():
        # print(i)
        # print(pt.all())
        # print(np.isfinite(pt).all())
        pt_x = int(pt[0][0])
        pt_y = int(pt[0][1])
        if pt_x > 0 and pt_y > 0:
            # use the BGR format to match the original image type
            col = (int(colors[i, 2]), int(colors[i, 1]), int(colors[i, 0]))
            cv2.circle(blank_img, (pt_x, pt_y), 1, col)


cv2.imshow('colorized',blank_img)
cv2.waitKey()
# im.set_data(depth)
# plt.hist(depth)
# plt.show()

depth_thresh = 1.2 # Threshold for SAFE distance (in cm)
 
# Mask to segment regions with depth less than threshold
mask = cv2.inRange(depth,0.1,depth_thresh)

cv2.imshow('depth mask',mask)
cv2.waitKey()
 
# Check if a significantly large obstacle is present and filter out smaller noisy regions
if np.sum(mask)/255.0 > 0.01*mask.shape[0]*mask.shape[1]:
 
  # Contour detection 
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cnts = sorted(contours, key=cv2.contourArea, reverse=True)
   
  # Check if detected contour is significantly large (to avoid multiple tiny regions)
  if cv2.contourArea(cnts[0]) > 0.01*mask.shape[0]*mask.shape[1]:
 
    x,y,w,h = cv2.boundingRect(cnts[0])
 
    # finding average depth of region represented by the largest contour 
    mask2 = np.zeros_like(mask)
    cv2.drawContours(mask2, cnts, 0, (255), -1)
    cv2.drawContours(fixedLeft, cnts, 0, (255), -1)
    # Calculating the average depth of the object closer than the safe distance
    depth_mean, _ = cv2.meanStdDev(depth, mask=mask2)
     
    # Display warning text
    cv2.putText(fixedLeft, "WARNING !", (x+5,y-40), 1, 2, (0,0,255), 2, 2)
    cv2.putText(fixedLeft, "Object at", (x+5,y), 1, 2, (100,10,25), 2, 2)
    cv2.putText(fixedLeft, "%.2f cm"%depth_mean, (x+5,y+40), 1, 2, (100,10,25), 2, 2)
 
else:
  cv2.putText(fixedLeft, "SAFE!", (100,100),1,3,(0,255,0),2,3)
 
cv2.imshow('output_canvas',fixedLeft)
cv2.waitKey()