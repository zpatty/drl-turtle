import numpy as np
import cv2 as cv
import os

# first half of this file is attempting to find the camera matrix and distortion coefficients of the two cameras
# the second half is calibrating them together to try to build a depth map

## updates/todo:

# try out code with new [ZED Mini] camera (current code is for ELP camera)
# will need to find the camera intrinsics matrix, possibly using MATLAB
# correct depth map until seeing a reasonable result (possibly fiddling with params/algos/flags)

#calibration pattern, chessboard is 7x10 (internal corners)
row_corners=7
col_corners=10
square_unit=1
#how many units is the side of one square
square_dim_mm= 23 
square_dim_cm= 2.3
square_dim_in= 29/32
pattern_size=(col_corners, row_corners)
count=row_corners*col_corners
image_size=(640,360) # important to make sure this is correct

#goes through the folder of sample images, and adds paths to the images
#to a list, and also counts the number of images in the folder

#make image finding a function of the folder
#probably nicer to return as a n lenght list of 3 tuples, for n images
#where find_corners(folder)[i] is the (image, pattern, corners) for the 
#ith image, and then num_images is the len of this list

def find_corners(folder):
    #param: folder is the string of the name of the folder of images you are searching
    #runs find chessboard corners function on the images in the folder
    #returns: list containing a tuple (img, find_chess[0], find_chess[1]) for each image 
    #where img is the read in image 
    #find_chess[0] is the bool of whether the pattern was found or not
    #and find_chess[1] is either None if no pattern found, or the array of corners
    result=[]
    image_names=generate_filenames(folder)
    images=read_images(image_names, folder)
    flags=cv.CALIB_CB_ADAPTIVE_THRESH
    corners=np.empty((col_corners, row_corners))
    for img in images:
        find_chess=cv.findChessboardCorners(img, pattern_size, corners, flags)
        result.append((img, find_chess[0], find_chess[1]))
    return result

def find_corners_old(folder):
    #param: folder is the string of the name of the folder of images you are searching
    #runs find chessboard corners function on the images in the folder
    #returns: tuple containing list of all images read, num of images
    #and findchessboardcorners results; pattern found and corners found
    #pattern found is a list of bools on whether the pattern was found or not
    #corners found is a list of either None if no pattern was found, or an array of corners found
    image_names=generate_filenames(folder)
    images=read_images(image_names, folder)
    num_images=len(image_names)
    flags=cv.CALIB_CB_ADAPTIVE_THRESH
    pattern_found=[]
    corners_found=[]
    corners=np.empty((col_corners, row_corners))
    for img in images:
        find_chess=cv.findChessboardCorners(img, pattern_size, corners, flags)
        pattern_found.append(find_chess[0])
        #find_chess[0] is a bool of if the pattern was found or not
        if np.any(find_chess[1])==None:
            corners_found.append([])
        else:
            corners_found.append(find_chess[1])
        #find_chess[1] is either None if no pattern was found, or an array
    return (images, num_images, pattern_found, corners_found)

def generate_filenames(folder):
    #takes in the name of a folder (str)
    # returns a list of file names in the folder
    image_names=[]
    for image in os.walk(folder):
        image_names=image[2]
    #remove invisible os folder
    if '.DS_Store' in image_names:
        image_names.remove('.DS_Store')
    return image_names

def read_images(image_names, folder):
    #takes in a list of file names and the name of the folder (str) they're in 
    #returns list of read in images
    images=[]
    for image in image_names:
        #img_str='calibration_images/'+image
        img_str=folder+'/'+image
        img_path=os.path.abspath(img_str)
        grey_image=cv.cvtColor(cv.imread(img_path),cv.COLOR_BGR2GRAY)
        images.append(grey_image)
    return images

def convert_color(img):
    #takes in a black and white image and returns it as a 
    #three channel image
    grey=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    color_img = np.zeros_like(img)
    for n in range(3):
        color_img[:,:,n] = grey
    return color_img

def no_pattern(pattern_found):
    #takes in a list of bools, whether a pattern was found
    #returns indices of images where no pattern was found
    indices=set()
    for i in range(len(pattern_found)):
        if not pattern_found[i]:
            indices.add(i)
    return indices

# getting image points is worse in the new return version 
left_img=find_corners_old('new_imgs/left')
right_img=find_corners_old('new_imgs/right')

#find_corners_old(folder)[0] is a list of images read by cv
L_images=left_img[0]
R_images=right_img[0]

#find_corners_old(folder)[1] is the number of images in the folder
num_L=left_img[1]
num_R=right_img[1]
if num_L==num_R:
    LR_num=num_L
else:
    print("left and right num images mismatch")

#find_corners_old(folder)[2] is a list of bools (the bools are gen by find_chess[0])
L_pat_found=left_img[2]
R_pat_found=right_img[2]

#find_corners_old(folder)[3] is a list of:
# if pattern found, length 70 array where each element is a list of a list of 2 floats 
# if pattern not found, empty list
L_cor_found=left_img[3]
R_cor_found=right_img[3]

to_remove=no_pattern(L_pat_found).union(no_pattern(R_pat_found))

#we want a list of images and then their corners
L_new=[[],[]]
R_new=[[],[]]
for i in range(LR_num):
    if i not in to_remove:
        L_new[0].append(L_images[i])
        L_new[1].append(L_cor_found[i])
        R_new[0].append(R_images[i])
        R_new[1].append(R_cor_found[i])

def get_subpix_corns(images, corners):
    sub_corners=[]
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 5, 1)
    win=(7,7)
    zero_zone=(-1,-1)
    for i in range(len(images)):
        sub_corners.append(cv.cornerSubPix(images[i], corners[5], win, zero_zone, criteria))
    return sub_corners

L_sub_corn=get_subpix_corns(L_new[0],L_new[1])
R_sub_corn=get_subpix_corns(R_new[0],R_new[1])

if len(L_new[0])==len(R_new[0]):
    LR_num=len(L_new[0])
    print("LR_num:"+str(LR_num))
else:
    print("left and right num new images mismatch")


#N is the number of total points being processed (aka # points per image * num images)
single_n=count*LR_num

L_img_pts=np.float32([np.reshape(L_sub_corn,(single_n,2))])
R_img_pts=np.float32([np.reshape(R_sub_corn,(single_n,2))])


#find corners debugging using draw corners below

# img=images[1]
# color_img=convert_color(img)
# draw=cv.drawChessboardCorners(color_img, pattern_size, corners_found[1], pattern_found[1])
# cv.imshow('drawn_corners', draw)
# k = cv.waitKey(0)


#create object points array 
# should be N by 3
#use obj_pts_build to create the values
#reshape and store as obj_pts

obj_pts_build=np.zeros((count,3),  np.float32)
obj_pts_build[:, :2]=np.mgrid[0:col_corners,0:row_corners].T.reshape(-1,2)
obj_pts_build*=square_dim_cm
obj_pts=[]
for i in range(LR_num):
    obj_pts.append(obj_pts_build)

obj_pts=np.float32([(np.reshape(obj_pts,(single_n,3)))])

# for i in range(single_n):
#         for x in range(row_corners):
#             for y in range(col_corners):
                
#                 obj_pts_build[i]=[x*square_dim_cm,y*square_dim_cm,0]
# print(len(obj_pts))
# print("object points"+str(obj_pts))

#for i in range(single_n):
#obj_pts_build[i]=(x*square_dim_cm,y*square_dim_cm,0)

#should contain vector of vectors of points of type Point3f 

# print(obj_pts_build, np.ndim(obj_pts_build),np.shape(obj_pts_build),np.size(obj_pts_build))
# print(len(L_img_pts))
# print("image points"+str(L_img_pts))


int_mat=[]
dist_coef=[]
# print('OBJPTS'+str(len(obj_pts_build)))
# print('IMGPTS'+str(len(left_img[3])))
# print('PTCOUNT'+str(len(point_count)))
# print('IMGSIZE'+str(len(image_size)))
calib_left=cv.calibrateCamera(obj_pts, L_img_pts, image_size, None, None)
calib_right=cv.calibrateCamera(obj_pts, R_img_pts, image_size, None, None)

# print("calibrate left:")
# print(calib_left)
# print("calibrate right:")
# print(calib_right)

# L:
# (19.506522211398817, np.array([[4.73138213e+03, 0.00000000e+00, 5.18223988e+02],
#        [0.00000000e+00, 3.57730927e+03, 1.50282704e+02],
#        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), np.array([[ 2.22437643e+02, -3.08950012e+05, -1.03720358e+00,
#         -1.67260986e+00, -7.33926298e+02]]), (np.array([[-0.13170359],
#        [ 0.72622837],
#        [-0.51533282]]),), (np.array([[-12.56764548],
#        [ -6.10354988],
#        [369.48507841]]),))
# R:
# (20.886516351459832, np.array([[532.10490591,   0.        , 505.56578522],
#        [  0.        , 515.16229963, 182.48782071],
#        [  0.        ,   0.        ,   1.        ]]), np.array([[ 1.53890218e+01, -3.07141685e+02,  2.20949195e-02,
#         -5.44633314e-01,  1.41157321e+03]]), (np.array([[ 0.10879618],
#        [ 0.51182915],
#        [-0.11120764]]),), (np.array([[-15.0980348 ],
#        [-12.35261895],
#        [ 56.67723485]]),))

left_cam_matrix=calib_left[1]
left_dist_coeff=calib_left[2]
right_cam_matrix=calib_right[1]
right_dist_coeff=calib_right[2]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CAMERA CALIBRATION ATTEMPTS ABOVE, STEREO CALIBRATION BELOW USING MATLAB RESULTS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Camera intrinsics matrices take the form [[f_x,0,c_x],[0,f_y,c_y],[0,0,1]]
# where f_x and f_y are the actual physical focal lengths of the lens
# and c_x and c_y model the displacement of the center of coordinates away from the optic axis to the projection screen
#Displacement coefficients are listed as a vector [k_1, k_2, p_1, p_2, k_3] 
# where k_1, k_2 & k_3 are radial distortion, with k_3 usually only used for fish-eye lenses 
# and p_1 and p_2 are tangential distortion

#From MATLAB, got the following matrices:
L_cam_mat= np.array([[422.313321161968,0,345.824901119105],[
0,422.692523040881,260.071545477633],[
0,0,1]])
R_cam_mat= np.array([[422.104286896807,0,337.316046510253],[
0,422.354660812964,272.064383556624
],[0,0,1]])

L_dist_coeff=np.array([-0.388717236012473,0.142441811337318,0,0,0])
R_dist_coeff=np.array([-0.389481758183715,0.142311269649709,0,0,0])
#^ note that only k_1 and k_2 are being used 

# img=R_new[0][1]

# opt_L_matrix=cv.getOptimalNewCameraMatrix(left_cam_matrix,left_dist_coeff,image_size)

# undist_map=cv.initUndistortRectifyMap(R_cam_mat,R_dist_coeff,np.identity(3),R_cam_mat,image_size,cv.CV_32FC1)
# undist_img=cv.remap(img,undist_map[0],undist_map[1],cv.INTER_NEAREST)

# cv.imshow('og_img', img)
# cv.imshow('undist_img', undist_img)
# k = cv.waitKey(0)

# ---unused code for when we get errors about shape of obj/img pts---
# L_img_pts=np.float32(np.reshape(L_img_pts,(count,LR_num,2)))
# R_img_pts=np.float32(np.reshape(R_img_pts,(count,LR_num,2)))
# obj_pts_build=np.float32(np.reshape(obj_pts_build,(70,3)))
          
# print((obj_pts).shape)
# print((L_img_pts).shape)
# print((R_img_pts).shape)
# error resolved by using obj_pts instead of obj_pts_build
# -------------------------------------------------------------------

stereo_output=cv.stereoCalibrate(obj_pts,L_img_pts,R_img_pts,L_cam_mat,L_dist_coeff,R_cam_mat,R_dist_coeff,image_size)

# for i in range(len(stereo_output)):
#     print("STEREO RESULT NUM "+str(i)+" : "+str(stereo_output[i]))

#stereoCalibrate returns an array of 9 values: 
# 0: retval (aka error)
# 1: camera matrix 1 
# 2: dist. coeff 1 
# 3: camera matrix 2 
# 4: dist coeff 2 
# 5: R the rotation matrix 
# 6: T the translation matrix
# 7: E the essential matrix 
# 8: F the fundamental matrix

rot_mat=stereo_output[5]
trans_mat=stereo_output[6]
ess_mat=stereo_output[7]
fund_mat=stereo_output[8]

stereo_rect=cv.stereoRectify(L_cam_mat,L_dist_coeff,R_cam_mat,R_dist_coeff,image_size, rot_mat,trans_mat, None,None,None,None,None, cv.CALIB_ZERO_DISPARITY,.25)

# for i in range(len(stereo_rect)):
#      print("STEREO RECTIFY RESULT NUM "+str(i)+": "+str(stereo_rect[i]))

# STEREO RECTIFY RESULT NUM 0: [[ 0.88385679  0.2626599   0.38704903]
#  [-0.29164507  0.95637594  0.01697682]
#  [-0.36570525 -0.12788602  0.92190283]]
# STEREO RECTIFY RESULT NUM 1: [[ 0.93813861  0.33087347  0.10207202]
#  [-0.32218272  0.94211986 -0.09278178]
#  [-0.12686311  0.05415633  0.99044073]]
# STEREO RECTIFY RESULT NUM 2: [[818.92051326   0.         155.52271843   0.        ]
#  [  0.         818.92051326 335.5461998    0.        ]
#  [  0.           0.           1.           0.        ]]
# STEREO RECTIFY RESULT NUM 3: [[ 8.18920513e+02  0.00000000e+00  1.55522718e+02 -8.78728006e+03]
#  [ 0.00000000e+00  8.18920513e+02  3.35546200e+02  0.00000000e+00]
#  [ 0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00]]
# STEREO RECTIFY RESULT NUM 4: [[ 1.00000000e+00  0.00000000e+00  0.00000000e+00 -1.55522718e+02]
#  [ 0.00000000e+00  1.00000000e+00  0.00000000e+00 -3.35546200e+02]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  8.18920513e+02]
#  [ 0.00000000e+00  0.00000000e+00  9.31938561e-02 -0.00000000e+00]]
# STEREO RECTIFY RESULT NUM 5: (0, 81, 640, 279)
# STEREO RECTIFY RESULT NUM 6: (0, 0, 640, 207)

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
L_undist_map=cv.initUndistortRectifyMap(L_cam_mat,L_dist_coeff,np.identity(3),L_cam_mat,image_size,cv.CV_32FC1)
R_undist_map=cv.initUndistortRectifyMap(R_cam_mat,R_dist_coeff,np.identity(3),R_cam_mat,image_size,cv.CV_32FC1)
# undist_img=cv.remap(img,undist_map[0],undist_map[1],cv.INTER_NEAREST)

#cv.reprojectImageTo3D()
left_maps=cv.initUndistortRectifyMap(L_cam_mat,L_dist_coeff,L_rect_trans,L_proj_mat,image_size,cv.CV_32FC1)
right_maps=cv.initUndistortRectifyMap(R_cam_mat,R_dist_coeff,R_rect_trans,R_proj_mat,image_size,cv.CV_32FC1)

L_map_x=left_maps[0]
L_map_y=left_maps[1]
R_map_x=right_maps[0]
R_map_y=right_maps[1]

#the 16 & 15 picked because opencv's sample code used it :/
stereo=cv.StereoBM_create(16,15)
#someone else's code I looked at has all this junk, but i think it makes things worse?
# stereo.setMinDisparity(4)
# stereo.setNumDisparities(128)
# stereo.setBlockSize(21)
# stereo.setROI1(roi1)
# stereo.setROI2(roi2)
# stereo.setSpeckleRange(16)
# stereo.setSpeckleWindowSize(45)

cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FPS, 120)

cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 360)
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

#left cam has worse distortion than right, also the unrectified version 
# is better abt the cropping issue but worse about the distortion
#also does a weird left shift thing
#also the depth map is pretty borked

#  #params: objpts, imgpts, ptcounts,image_size
# #intrinsic matrix, dist.coef, rotvec=null, transvec=null, flags=0
# #opts,imgpts see below, pt_count= # pts, as a Mx1 matrix
# #image size is size in pixels, of the images
# #pg 392
# opts = N x 3 matrix
# N=K(num pts= count) * M (num images=1/2 tot_num_images)
# ^ basically the point coords in whatever unit we want it to be 
# if a real unit, need to measure the squares

# ipts = N x 2 matrix
# pixel coords, supply from find chessboard corners

# point_counts= num pts in the image M x 1 matrix
# image_size = pixel size of the image_size
# instrinstic_matrix & dist coef can be outputs


# testing

# img=left_img[0][15]
# pattern=left_img[2][15]
# corners=left_img[3][15]
# color_img=convert_color(img)
# draw=cv.drawChessboardCorners(color_img, pattern_size, corners, pattern)
# cv.imshow('drawn_corners', draw)
# k = cv.waitKey(0)
