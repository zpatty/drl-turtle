import numpy as np
import cv2 as cv
import os

## updates/todo:

# -> obj pts error resolved:
# "objectPoints should contain vector of vectors of points of type Point3f"
# the np arrays, obj_pts and img_pts wanted to be lists inside np arrays
# so I changed that and now cv.calibrateCamera runs!

# -> testing cv.calibrateCamera results by running
# cv.initUndistortRectifyMap and cv.remap to apply the 
# camera matrix and distortion coeff found from cv.calibrateCamera
# current result has worse distrotion than the original :(
# but this is a known issue (well hopefully its the same issue)
# and I should try cropping the images to 960x720
# may have to retake sample images if the chessboard is affected

#calibration pattern, chessboard is 7x10 (internal corners)
row_corners=7
col_corners=10
square_unit=1
#how many units is the side of one square
square_dim_mm= 23 
square_dim_cm= 2.3
square_dim_in= 29/32
pattern_size=(row_corners,col_corners)
count=row_corners*col_corners
image_size=(1280,360)

#goes through the folder of sample images, and adds paths to the images
#to a list, and also counts the number of images in the folder

#make image finding a function of the folder
#probably nicer to return as a n lenght list of 3 tuples, for n images
#where find_corners(folder)[i] is the (image, pattern, corners) for the 
#ith image, and then num_images is the len of this list

def find_corners(folder):
    #param: folder is the string of the name of 
    #the folder of images you are searching
    #runs find chessboard corners function on the images in the folder
    #returns: tuple containing list of all images read, num of images
    #and findchessboardcorners results; pattern found and corners found

    result=[]
    image_names=generate_filenames(folder)
    images=read_images(image_names, folder)

    flags=cv.CALIB_CB_ADAPTIVE_THRESH

    #given images of the chessboard at different 
    #positions/angles in space, find the corners

    corners=np.empty((row_corners,col_corners))
    for img in images:
        find_chess=cv.findChessboardCorners(img, pattern_size, corners, flags)
        result.append((img, find_chess[0], find_chess[1]))

    # return(images, num_images, pattern_found, corners_found)
    return result

def find_corners_old(folder):
    #param: folder is the string of the name of 
    #the folder of images you are searching
    #runs find chessboard corners function on the images in the folder
    #returns: tuple containing list of all images read, num of images
    #and findchessboardcorners results; pattern found and corners found

    image_names=generate_filenames(folder)
    images=read_images(image_names, folder)
    num_images=len(image_names)
    
    flags=cv.CALIB_CB_ADAPTIVE_THRESH

    #given images of the chessboard at different 
    #positions/angles in space, find the corners
    pattern_found=[]
    corners_found=[]
    corners=np.empty((row_corners,col_corners))
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
left_img=find_corners_old('calib_img/left')
right_img=find_corners_old('calib_img/right')

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
else:
    print("left and right num new images mismatch")


#N is the number of total points being processed (aka # points per image * num images)
single_n=count*LR_num

L_img_pts=np.float32([np.reshape(L_sub_corn,(single_n,2))])
R_img_pts=np.float32([np.reshape(R_sub_corn,(single_n,2))])

#find corners debugging using draw corners

# img=images[1]
# color_img=convert_color(img)
# draw=cv.drawChessboardCorners(color_img, pattern_size, corners_found[1], pattern_found[1])
# cv.imshow('drawn_corners', draw)
# k = cv.waitKey(0)


#create object points array 
# should be N by 3

#doesn't currently account for real world dimensions

obj_pts=np.zeros((count,3),  np.float32)
obj_pts[:, :2]=np.mgrid[0:row_corners,0:col_corners].T.reshape(-1,2)
obj_pts*=square_dim_cm
new_obj_pts=[]
for i in range(LR_num):
    new_obj_pts.append(obj_pts)

new_obj_pts=np.float32([(np.reshape(new_obj_pts,(single_n,3)))])

# for i in range(single_n):
#         for x in range(row_corners):
#             for y in range(col_corners):
                
#                 obj_pts[i]=[x*square_dim_cm,y*square_dim_cm,0]
# print(len(new_obj_pts))
# print("object points"+str(new_obj_pts))

#for i in range(single_n):
#obj_pts[i]=(x*square_dim_cm,y*square_dim_cm,0)

#should contain vector of vectors of points of type Point3f 

# print(obj_pts, np.ndim(obj_pts),np.shape(obj_pts),np.size(obj_pts))
# print(len(L_img_pts))
# print("image points"+str(L_img_pts))

int_mat=[]
dist_coef=[]
# print('OBJPTS'+str(len(obj_pts)))
# print('IMGPTS'+str(len(left_img[3])))
# print('PTCOUNT'+str(len(point_count)))
# print('IMGSIZE'+str(len(image_size)))
print("calibrate left:")
print(cv.calibrateCamera(new_obj_pts, L_img_pts, image_size, None, None))
print("calibrate right:")
print(cv.calibrateCamera(new_obj_pts, R_img_pts, image_size, None, None))

calib_left=(19.506522211398817, np.array([[4.73138213e+03, 0.00000000e+00, 5.18223988e+02],
       [0.00000000e+00, 3.57730927e+03, 1.50282704e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), np.array([[ 2.22437643e+02, -3.08950012e+05, -1.03720358e+00,
        -1.67260986e+00, -7.33926298e+02]]), (np.array([[-0.13170359],
       [ 0.72622837],
       [-0.51533282]]),), (np.array([[-12.56764548],
       [ -6.10354988],
       [369.48507841]]),))
calib_right=(20.886516351459832, np.array([[532.10490591,   0.        , 505.56578522],
       [  0.        , 515.16229963, 182.48782071],
       [  0.        ,   0.        ,   1.        ]]), np.array([[ 1.53890218e+01, -3.07141685e+02,  2.20949195e-02,
        -5.44633314e-01,  1.41157321e+03]]), (np.array([[ 0.10879618],
       [ 0.51182915],
       [-0.11120764]]),), (np.array([[-15.0980348 ],
       [-12.35261895],
       [ 56.67723485]]),))

left_cam_matrix=calib_left[1]
left_dist_coeff=calib_left[2]
right_cam_matrix=calib_right[1]
right_dist_coeff=calib_right[2]

img=L_new[0][1]

undist_map=cv.initUndistortRectifyMap(left_cam_matrix,left_dist_coeff,np.identity(3),left_cam_matrix,image_size,cv.CV_32FC1)
undist_img=cv.remap(img,undist_map[0],undist_map[1],cv.INTER_NEAREST)

cv.imshow('og_img', img)
cv.imshow('undist_img', undist_img)
k = cv.waitKey(0)


#current error: objectPoints should contain 
# vector of vectors of points of type Point3f in function 'collectCalibrationData'

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
