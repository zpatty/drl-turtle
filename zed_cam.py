import numpy as np
import cv2 as cv
import os

## updates/todo:

# try out code with new [ZED Mini] camera (current code is for ELP camera)
# will need to find the camera intrinsics matrix, possibly using MATLAB
# correct depth map until seeing a reasonable result (possibly fiddling with params/algos/flags)

IMAGE_SIZE=(1280,720) #cropping og screen in half as normally the left and right are displayed side by side

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ZED CAMERA PARAMETERS, GIVEN CALIBRATION 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
M_L=np.array([[773.73,0,635.5],[
0,773.535,371.6],[
0,0,1]]) #left instrinics camera matrix
DIST_L=np.array([-0.034939,0.0220154,0,0,-0.0100592]) #left distortion vector
M_R=np.array([[772.175,0,640.49],[
0,771.92,359.354],[
0,0,1]])  #right instrinics camera matrix
DIST_R=np.array([-0.0340837,0.0154457,0,0,0.00337671]) #right distortion vector

CAM_DIST=62.9812 # distance btwn the two cameras, unsure where to use it but i think it should be useful??
T_X=0 #translation coefficients 
T_Y=0.0102655 #translation coefficients 
T_Z=0.322348 #translation coefficients 
R_X=0.00158478 #rotation coefficients 
R_Y=0.00259672 #rotation coefficients  
R_Z=0.0011051 #rotation coefficients 

rot=cv.Rodrigues(np.array([R_X,R_Y,R_Z]),None,None)
R=rot[0]
T=np.array([[T_X],[T_Y],[T_Z]]) #translation matrix
S=np.array([[0,-T_Z,T_Y],[T_Z,0,-T_X],[-T_Y,T_X,0]]) #rotation matrix
E=np.matmul(R,S) #essential matrix (not sure if useful?)
F=np.matmul(np.transpose(np.linalg.inv(M_R)),np.matmul(E,np.linalg.inv(M_L))) #fundamental matrix (not sure if useful?)

# print("M_L"+str(M_L))
# print("DIST_L"+str(DIST_L))
# print("M_R"+str(M_R))
# print("DIST_R"+str(DIST_R))
# print("IMAGE_SIZE"+str(IMAGE_SIZE))
# print("R"+str(R))
# print("T"+str(T))

# (leftRectification, rightRectification, leftProjection, rightProjection,
#         dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
#                 leftCameraMatrix, leftDistortionCoefficients,
#                 rightCameraMatrix, rightDistortionCoefficients,
#                 imageSize, rotationMatrix, translationVector,
#                 None, None, None, None, None,
#                 cv2.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)

stereo_rect=cv.stereoRectify(M_L,DIST_L,M_R,DIST_R,IMAGE_SIZE,R,T,None,None,None,None,None, cv.CALIB_ZERO_DISPARITY,.25)

# for i in range(len(stereo_rect)):
#      print("STEREO RECTIFY RESULT NUM "+str(i)+": "+str(stereo_rect[i]))

# 0: 3x3 rectification transform (rot mat) for camera 1
# 1: 3x3 rectification transform (rot mat) for camera 2
# 2: 3x4 projection matrix in the new (rectified) coordinate systems for cam 1
# 3: 3x4 projection matrix in the new (rectified) coordinate systems for cam 2
# 4: 4x4 disparity-to-depth mapping matrix 
# 5 & 6: roi1, roi2 rectangles inside the rectified images where all the pixels are valid [x,y,w,h]

L_rect_trans=stereo_rect[0]
R_rect_trans=stereo_rect[1]
L_proj_mat=stereo_rect[2]
R_proj_mat=stereo_rect[3]
disp_to_depth=stereo_rect[4]
roi1=stereo_rect[5]
roi2=stereo_rect[6]

print("ROI1:"+str(roi1))
print("ROI2:"+str(roi2))


#now that we have stereo calibration values, we want to apply them to the images
L_undist_map=cv.initUndistortRectifyMap(M_L,DIST_L,np.identity(3),M_L,IMAGE_SIZE,cv.CV_32FC1)
R_undist_map=cv.initUndistortRectifyMap(M_R,DIST_R,np.identity(3),M_R,IMAGE_SIZE,cv.CV_32FC1)
# undist_img=cv.remap(img,undist_map[0],undist_map[1],cv.INTER_NEAREST)

#cv.reprojectImageTo3D()
left_maps=cv.initUndistortRectifyMap(M_L,DIST_L,L_rect_trans,L_proj_mat,IMAGE_SIZE,cv.CV_32FC1)
right_maps=cv.initUndistortRectifyMap(M_R,DIST_R,R_rect_trans,R_proj_mat,IMAGE_SIZE,cv.CV_32FC1)

L_map_x=left_maps[0]
L_map_y=left_maps[1]
R_map_x=right_maps[0]
R_map_y=right_maps[1]

#the 16 & 15 picked because opencv's sample code used it :/
stereo=cv.StereoBM_create(16,15)

stereo.setPreFilterSize(41)
stereo.setPreFilterCap(31)
stereo.setSpeckleWindowSize(41)
stereo.setMinDisparity(-16)
stereo.setNumDisparities(128)
stereo.setTextureThreshold(10)
stereo.setUniquenessRatio(15)
#someone else's code I looked at has all this junk, but i think it makes things worse?
# stereo.setMinDisparity(4)
# stereo.setNumDisparities(64)
# stereo.setBlockSize(21)
# stereo.setROI1(roi1)
# stereo.setROI2(roi2)
# stereo.setSpeckleRange(16)
stereo.setSpeckleWindowSize(40)

cam = cv.VideoCapture(1)
cam.set(cv.CAP_PROP_FPS, 120)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 2560)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
s,orignal = cam.read()
height, width, channels = orignal.shape
print(width)
print(height)


while(True):
    s,orignal = cam.read()
    left=orignal[0:height,0:int(width/2)]
    right=orignal[0:height,int(width/2):(width)]
    
    if not s:
        print("failed to grab frame")
        break


    fixedLeft = cv.remap(left, L_undist_map[0], L_undist_map[1], cv.INTER_LINEAR)
    fixedRight = cv.remap(right, R_undist_map[0], R_undist_map[1], cv.INTER_LINEAR)

    # fixedLeft = cv.remap(left, L_map_x, L_map_y, cv.INTER_LINEAR)
    # fixedRight = cv.remap(right, R_map_x, R_map_y, cv.INTER_LINEAR)

    grayLeft = cv.cvtColor(fixedLeft, cv.COLOR_BGR2GRAY)
    grayRight = cv.cvtColor(fixedRight, cv.COLOR_BGR2GRAY)
    depth = stereo.compute(grayLeft, grayRight)

    cv.imshow('left', fixedLeft)
    cv.imshow('right', fixedRight)
    cv.imshow('depth', depth/2048)
    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()
cv.destroyAllWindows()