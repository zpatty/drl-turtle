import numpy as np
import cv2 as cv
import os

# s=cv.imread(path)
# cv.imshow('test', s)
# k = cv.waitKey(0)

#what do we do if it can't find the corners
# -> cut both the left and right images from our data set?
# if so, should take more images

#######

row_corners=7
col_corners=10
square_unit=1
square_dim_mm= 23 #how many units is the side of one square
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
    image_names=[]
    for image in os.walk(folder):
        image_names=image[2]
    #remove invisible os folder
    if '.DS_Store' in image_names:
        image_names.remove('.DS_Store')

    images=[]
    for image in image_names:
        #img_str='calibration_images/'+image
        img_str=folder+'/'+image
        img_path=os.path.abspath(img_str)
        images.append(cv.imread(img_path))

    flags=cv.CALIB_CB_ADAPTIVE_THRESH

    #given images of the chessboard at different 
    #positions/angles in space, find the corners

    #want to split into two, left vs right
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

    image_names=[]
    for image in os.walk(folder):
        image_names=image[2]
    #remove invisible os folder
    if '.DS_Store' in image_names:
        image_names.remove('.DS_Store')
    num_images=len(image_names)

    images=[]
    for image in image_names:
        #img_str='calibration_images/'+image
        img_str=folder+'/'+image
        img_path=os.path.abspath(img_str)
        images.append(cv.imread(img_path))
    
    flags=cv.CALIB_CB_ADAPTIVE_THRESH

    #given images of the chessboard at different 
    #positions/angles in space, find the corners
    pattern_found=[]
    corners_found=[]
    #want to split into two, left vs right
    corners=np.empty((row_corners,col_corners))
    for img in images:
        find_chess=cv.findChessboardCorners(img, pattern_size, corners, flags)
        pattern_found.append(find_chess[0])
        corners_found.append(find_chess[1])

    return(images, num_images, pattern_found, corners_found)

def convert_color(img):
    #takes in a black and white image and returns it as a 
    #three channel image
    grey=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    color_img = np.zeros_like(img)
    for n in range(3):
        color_img[:,:,n] = grey
    return color_img

def LR_num_check(l,r):
    if l != r:
        print("left and right num images mismatch")

# calib_img=find_corners('calibration_images')
# left_img=find_corners_old('calib_img/left')
# right_img=find_corners('calib_img/right')

# l_img=left_img[0]
# l_num_img=left_img[1]
# l_pattern=left_img[2]
# l_corners=left_img[3]
# r_img=right_img[0]
# r_num_img=right_img[1]
# r_pattern=right_img[2]
# r_corners=right_img[3]

# num_left=left_img[1]
# print(num_left)
# num_right=len(right_img)
# LR_num_check(num_left,num_right)

# l_recognized_img=left_img.copy()
# r_recognized_img=right_img.copy()

# for i in range(num_left):
#     if not (left_img[i][1] and right_img[i][1]):
#         L_ind=l_recognized_img.index(np.all(left_img[i]))
#         R_ind=r_recognized_img.index(np.all(right_img[i]))
#         if L_ind != R_ind:
#             print("left and right indices unequal :/")
#         l_recognized_img.pop(L_ind)
#         r_recognized_img.pop(R_ind)
#         # L_ind=np.argwhere(l_recognized_img==left_img[i])
#         # R_ind=np.argwhere(r_recognized_img==right_img[i])
#         # if L_ind != R_ind:
#         #     print("left and right indices unequal :/")
#         # np.delete(l_recognized_img,L_ind)
#         # np.delete(r_recognized_img,R_ind)

# print('left right new recheck')
# LR_num_check(len(l_recognized_img),len(r_recognized_img))
# print('left new')
# print(l_recognized_img)
# print('right new')
# print(r_recognized_img)


# now try what if we dont get rid of the bad images, but 
# instead just acknowledge just some of the images will 
# get zero point count

# getting image points is worse in the new return version 
left_img=find_corners_old('calib_img/left')
right_img=find_corners_old('calib_img/right')
num_left=left_img[1]
num_right=right_img[1]
LR_num_check(num_left,num_right)
single_n=count*num_left
#something weird is happening here

# criteria=cv.TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 5, 1 )
# win=7 #not sure how to find this oops
# zero_zone=cv.size(-1,-1)
# subpix=cv.findCornerSubPix(image, corners, count, win, zero_zone, criteria)
# print(subpix)




# img=images[1]
# color_img=convert_color(img)
# draw=cv.drawChessboardCorners(color_img, pattern_size, corners_found[1], pattern_found[1])
# cv.imshow('drawn_corners', draw)
# k = cv.waitKey(0)

# single_n=count*single_num_img
# # print('count:'+str(count)+'single num img:'+str(single_num_img)+'n:'+str(single_n))
# #whats going on here

obj_pts=[]

for i in range(num_left):
    for x in range(row_corners):
        for y in range(col_corners):
            obj_pts.append([x*square_dim_cm,y*square_dim_cm,0])

obj_pts=np.reshape(obj_pts,(single_n,3))
# img_pts=np.reshape(corners_found, ())
# print(obj_pts, np.ndim(obj_pts),np.shape(obj_pts),np.size(obj_pts))

point_count=[]
for i in range(num_left):
    if left_img[2][i]:
        point_count.append(count)
    else:
        point_count.append(0)




#find chess analysis:
ex_result=(True, array([[[480.9781  , 143.34225 ]],

       [[481.4371  , 159.65384 ]],

       [[481.42267 , 178.04369 ]],

       [[481.83942 , 197.86916 ]],

       [[482.1807  , 217.70683 ]],

       [[481.9107  , 239.45363 ]],

       [[481.05258 , 261.07492 ]],

       [[462.17947 , 139.37393 ]],

       [[461.90216 , 156.2452  ]],

       [[461.86212 , 175.22809 ]],

       [[461.22818 , 193.53577 ]],

       [[460.9207  , 215.21472 ]],

       [[460.05273 , 235.65652 ]],

       [[459.0841  , 257.26187 ]],

       [[441.71634 , 135.42836 ]],

       [[440.71765 , 151.74854 ]],

       [[440.28192 , 170.60892 ]],

       [[439.58435 , 190.03362 ]],

       [[439.3392  , 210.50403 ]],

       [[437.35382 , 231.70042 ]],

       [[436.03696 , 254.75183 ]],

       [[420.71747 , 131.61197 ]],

       [[420.53497 , 148.65512 ]],

       [[419.10593 , 167.4501  ]],

       [[417.11954 , 186.23566 ]],

       [[415.95456 , 206.95787 ]],

       [[414.55746 , 228.67282 ]],

       [[412.38556 , 251.39316 ]],

       [[400.16895 , 128.1994  ]],

       [[399.31546 , 145.24507 ]],

       [[396.93597 , 163.47006 ]],

       [[395.30252 , 183.3717  ]],

       [[392.73816 , 203.36687 ]],

       [[391.30258 , 225.33041 ]],

       [[388.277   , 247.61395 ]],

       [[379.20267 , 125.31188 ]],

       [[376.74405 , 141.82202 ]],

       [[374.66702 , 159.88176 ]],

       [[372.22324 , 179.54276 ]],

       [[369.33105 , 199.613   ]],

       [[366.63623 , 221.64388 ]],

       [[363.57184 , 244.47348 ]],

       [[358.55936 , 122.27923 ]],

       [[355.53275 , 139.0253  ]],

       [[351.99942 , 157.34576 ]],

       [[349.51294 , 176.6644  ]],

       [[346.32608 , 196.84131 ]],

       [[342.95355 , 218.47177 ]],

       [[339.12277 , 241.30563 ]],

       [[336.7944  , 119.454575]],

       [[334.37366 , 136.12148 ]],

       [[329.348   , 153.9406  ]],

       [[326.5733  , 173.49292 ]],

       [[322.21912 , 193.6763  ]],

       [[318.9169  , 215.17314 ]],

       [[314.33667 , 237.62602 ]],

       [[316.21365 , 117.337296]],

       [[311.45468 , 133.89087 ]],

       [[308.3459  , 151.63705 ]],

       [[303.7173  , 170.85869 ]],

       [[299.81332 , 191.07976 ]],

       [[295.41953 , 212.32065 ]],

       [[290.5058  , 234.32616 ]],

       [[295.52893 , 115.077095]],

       [[291.35773 , 131.67152 ]],

       [[286.2486  , 149.64873 ]],

       [[281.4962  , 168.34421 ]],

       [[277.04355 , 188.42224 ]],

       [[271.8141  , 209.15773 ]],

       [[266.97018 , 231.47545 ]]], dtype=float32))

#so, we want to properly understand this and how we're using it
#its got two elements, a bool, and an array
# the array has two elements, a list and a type var
#the list has 70 elements, each of which is a list 
# of a list of two floats
# why is it 70?? -> the chessboard is 7 by 10, so each element is 
#the coord of a chessboard corner

#okay, and sometimes, instead of a list of corners, its None type 
# because no pattern was detected

x=[]
y=[]
# print(left_img[3][2])

# print(np.shape(left_img[3][2]))
# for coord in left_img[3][2]:
#     print(coord)
#     x.append(coord[0])
#     y.append(coord(1))
# new_pts=[x,y]

int_mat=[]
dist_coef=[]
# print('OBJPTS'+str(len(obj_pts)))
# print('IMGPTS'+str(len(left_img[3])))
# print('PTCOUNT'+str(len(point_count)))
# print('IMGSIZE'+str(len(image_size)))
# print(cv.calibrateCamera(obj_pts, new_pts, point_count, image_size, int_mat, dist_coef))

#getting the following error:
#cv2.error: OpenCV(4.7.0) :-1: error: (-5:Bad argument) in function 'calibrateCamera'
#> Overload resolution failed:
#>  - Can't parse 'imageSize'. Expected sequence length 2, got 51

#so, going to try reconfiguring img pts into a 2xN matrix instead, 
# as the results im printing are that only imgpts and ptcount are of 
# length 51, so unclear what is being referred to with 'imageSize'



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
