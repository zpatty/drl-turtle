import numpy as np
import cv2 as cv
import os

## updates/todo:
# > term criteria /subpixel corner finding resolved missing module issue:
# resolved, convert images to grey as they are being read in

# > img_pts:
# successfully removed images where no pattern is detected and applied 
# subpixel corner detection

#img points now works, problem that object points should be a vector 
# of vectors of points, according to error message

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

    image_names=generate_filenames(folder)
    images=read_images(image_names, folder)
    num_images=len(image_names)
    
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
        #find_chess[0] is a bool of if the pattern was found or not
        if np.any(find_chess[1])==None:
            corners_found.append([])
        else:
            corners_found.append(find_chess[1])
        #find_chess[1] is either None if no pattern was found, or an array

    return (images, num_images, pattern_found, corners_found)

def generate_filenames(folder):
    image_names=[]
    for image in os.walk(folder):
        image_names=image[2]
    #remove invisible os folder
    if '.DS_Store' in image_names:
        image_names.remove('.DS_Store')
    return image_names

def read_images(image_names, folder):
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

def LR_num_check(l,r):
    if l != r:
        print("left and right num images mismatch")

#the below code was trying to delete images with no 
# found pattern from the data, but didn't work bc numpy doesn't 
# like arrays where the subarrays are different sizes

def no_pattern(pattern_found):
    #returns indices of images where no pattern was found
    indices=set()
    for i in range(len(pattern_found)):
        if not pattern_found[i]:
            indices.add(i)
    return indices


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



print(no_pattern(L_pat_found), no_pattern(R_pat_found))
to_remove=no_pattern(L_pat_found).union(no_pattern(R_pat_found))
print(to_remove)

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

L_img_pts=np.reshape(L_sub_corn,(single_n,2))
R_img_pts=np.reshape(R_sub_corn,(single_n,2))
#find corners debugging using draw corners

# img=images[1]
# color_img=convert_color(img)
# draw=cv.drawChessboardCorners(color_img, pattern_size, corners_found[1], pattern_found[1])
# cv.imshow('drawn_corners', draw)
# k = cv.waitKey(0)


#create object points array

obj_pts=[]

for i in range(LR_num):
    for x in range(row_corners):
        for y in range(col_corners):
            obj_pts.append((x*square_dim_cm,y*square_dim_cm,0))

obj_pts=np.reshape(obj_pts,(single_n,3))

# img_pts=np.reshape(corners_found, ())
# print(obj_pts, np.ndim(obj_pts),np.shape(obj_pts),np.size(obj_pts))

#point count per image

# point_count=[]
# for i in range(LR_num):
#     if left_img[2][i]:
#         point_count.append(count)
#     else:
#         point_count.append(0)

int_mat=[]
dist_coef=[]
# print('OBJPTS'+str(len(obj_pts)))
# print('IMGPTS'+str(len(left_img[3])))
# print('PTCOUNT'+str(len(point_count)))
# print('IMGSIZE'+str(len(image_size)))
print(cv.calibrateCamera(obj_pts, L_img_pts, image_size, None, None))

#has a problem with image points, which is expected: saying item at [0] has wrong type,
# so probably the problem is that I have images which are not finding the pattern


#getting the following error:
#cv2.error: OpenCV(4.7.0) :-1: error: (-5:Bad argument) in function 'calibrateCamera'
#> Overload resolution failed:
#>  - Can't parse 'imageSize'. Expected sequence length 2, got 51

#so, going to try reconfiguring img pts into a 2xN matrix instead, 
# as the results im printing are that only imgpts and ptcount are of 
# length 51, so unclear what is being referred to with 'imageSize'

#x=[]
#y=[]
# print(left_img[3][2])

# print(np.shape(left_img[3][2]))
# for coord in left_img[3][2]:
#     print(coord)
#     x.append(coord[0])
#     y.append(coord(1))
# new_pts=[x,y]


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
