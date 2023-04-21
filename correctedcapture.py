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

ml_HD=np.array([[773.73,0,635.5],[
0,773.535,371.6],[
0,0,1]]) #left instrinics camera matrix
dl_HD=np.array([-0.034939,0.0220154,0,0,-0.0100592]) #left distortion vector
mr_HD=np.array([[772.175,0,640.49],[
0,771.92,359.354],[
0,0,1]])  #right instrinics camera matrix
dr_HD=np.array([-0.0340837,0.0154457,0,0,0.00337671]) #right distortion vector
HD=(ml_HD,dl_HD,mr_HD,dr_HD)

ml_VGA=np.array([[386.0875,0,335.745],[
0,385.96,187.177],[
0,0,1]]) #left instrinics camera matrix
dl_VGA=np.array([-0.0340837,0.0154457,0,0,0.00337671]) #left distortion vector
mr_VGA=np.array([[386.865,0,333.25],[
0,386.7675,193.3],[
0,0,1]])  #right instrinics camera matrix
dr_VGA=np.array([-0.034939,0.0220154,0,0,-0.0100592]) #right distortion vector
VGA=(ml_VGA,dl_VGA,mr_VGA,dr_VGA)

ml_FHD=np.array([[1544.35,0,963.98],[
0,1543.84,541.708],[
0,0,1]]) #left instrinics camera matrix
dl_FHD=np.array([-0.0340837,0.0154457,0,0,0.00337671]) #left distortion vector
mr_FHD=np.array([[1547.46,0,954],[
0,1547.07,566.2],[
0,0,1]])  #right instrinics camera matrix
dr_FHD=np.array([-0.034939,0.0220154,0,0,-0.01005921]) #right distortion vector
FHD=(ml_FHD,dl_FHD,mr_FHD,dr_FHD)

ml_2K=np.array([[1544.35,0,1107.98],[
0,1543.84,622.708],[
0,0,1]]) #left instrinics camera matrix
dl_2K=np.array([-0.0340837,0.0154457,0,0,0.00337671]) #left distortion vector
mr_2K=np.array([[1547.46,0,1098],[
0,1547.07,647.2],[
0,0,1]])  #right instrinics camera matrix
dr_2K=np.array([-0.034939,0.0220154,0,0,-0.01005921]) #right distortion vector
IN2K=(ml_2K,dl_2K,mr_2K,dr_2K)

#chosen_calib lets you chose which calibration parameters to use
#they seem to all have similar effects, except VGA, which is worse

chosen_calib=IN2K
for i in range(4):
    M_L=chosen_calib[0]
    DIST_L=chosen_calib[1]
      
    M_R=chosen_calib[2]
    DIST_R=chosen_calib[3]

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
# stereo.setSpeckleWindowSize(40)

cam = cv.VideoCapture(1)
cam.set(cv.CAP_PROP_FPS, 120)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 2560)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
s,orignal = cam.read()
height, width, channels = orignal.shape
print(width)
print(height)
img_counter = 0

#comparing corrected vs uncorrected images, currently correctly seems not very effective
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

    cv.imshow('og_left',left)

    cv.imshow('correct_left', fixedLeft)
    # cv.imshow('right', fixedRight)

    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif k%256 == 32:
        # SPACE pressed
        L_img_name = "0LZed_{}.png".format(img_counter)
        R_img_name = "0RZed_{}.png".format(img_counter)
        cv.imwrite(L_img_name, fixedLeft)
        cv.imwrite(R_img_name, fixedRight)
        print("{} written!".format(str(L_img_name)+' & '+str(R_img_name)))
        img_counter += 1


cam.release()
cv.destroyAllWindows()