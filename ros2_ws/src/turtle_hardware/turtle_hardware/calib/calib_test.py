import cv2
# assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from natsort import natsorted
import scipy


# If using with a list of filenames



DIM=(640, 480)

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

rightimages = natsorted(glob.glob('right/*.jpg'))
leftimages = natsorted(glob.glob('left/*.jpg'))
# print(rightimages)

KL=np.array([[708.3477312219868, 0.0, 260.69187590557686], [0.0, 675.3059166594338, 301.31936629865646], [0.0, 0.0, 1.0]])
DL=np.array([[-0.39383047117877457], [6.721465255404687], [-35.99917141986595], [61.49579122578909]])
KR=np.array([[667.0400978057647, 0.0, 334.8109094526051], [0.0, 644.922628956739, 364.07228200370565], [0.0, 0.0, 1.0]])
DR=np.array([[0.8809516193294453], [-6.609640306922403], [21.549513701823056], [-24.149385093847197]])
R=np.array([[0.8721459388442752, 0.02130940474841954, -0.4887815162490354], [-0.06589130347707366, 0.9950649291385254, -0.07418977641584823], [0.4847884048567273, 0.09691076342599622, 0.8692459412897253]])
T=np.array([[-2.085136618149882], [0.1939622251215522], [-0.9258137973647751]])*25.4
# leftimages = ['L_1.png', 'L_2.png', 'L_0.png', 'L_3.png', 'L_4.png', 'L_5.png', 'L_6.png', 'L_7.png', 'L_8.png', 'L_9.png', 'L_11.png', 'L_12.png', 'L_10.png', 'L_13.png', 'L_14.png', 'L_15.png', 'L_16.png', 'L_17.png', 'L_18.png', 'L_19.png', 'L_21.png', 'L_22.png', 'L_20.png', 'L_23.png', 'L_24.png', 'L_25.png', 'L_26.png', 'L_27.png', 'L_28.png', 'L_29.png', 'L_30.png', 'L_33.png', 'L_34.png', 'L_35.png', 'L_36.png', 'L_37.png', 'L_38.png', 'L_39.png', 'L_41.png', 'L_42.png', 'L_40.png', 'L_43.png', 'L_44.png', 'L_45.png', 'L_46.png', 'L_47.png', 'L_48.png', 'L_49.png', 'L_50.png', 'L_51.png', 'L_52.png']
# rightimages = ['R_1.png', 'R_2.png', 'R_0.png', 'R_3.png', 'R_4.png', 'R_5.png', 'R_6.png', 'R_7.png', 'R_8.png', 'R_9.png', 'R_11.png', 'R_12.png', 'R_10.png', 'R_13.png', 'R_14.png', 'R_15.png', 'R_16.png', 'R_17.png', 'R_18.png', 'R_19.png','R_21.png', 'R_22.png', 'R_20.png', 'R_23.png', 'R_24.png', 'R_25.png', 'R_26.png', 'R_27.png', 'R_28.png', 'R_29.png', 'R_30.png', 'R_33.png', 'R_34.png', 'R_35.png', 'R_36.png', 'R_37.png', 'R_38.png', 'R_39.png', 'R_41.png', 'R_42.png', 'R_40.png', 'R_43.png', 'R_44.png', 'R_45.png', 'R_46.png', 'R_47.png', 'R_48.png', 'R_49.png', 'R_50.png', 'R_51.png',  'R_52.png']

   


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

plt.ion()
fig, ax = plt.subplots()
first = 1
depth_mean_list = [10000000]
total_depth_list = [1000000]
filtered_total_depth = [1000000]
for i in range(1100, len(leftimages), 1):
  leftname=leftimages[i]
  rightname=rightimages[i]

  left = cv2.imread(leftname)
  right=cv2.imread(rightname)

  fixedLeft = cv2.remap(left, left1, left2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
  fixedRight = cv2.remap(right, right1, right2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
  # cv.imshow("fixedLeft", fixedLeft)
  # cv.imshow("fixedRight", fixedRight)

  grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
  grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
  disparity = stereo.compute(grayLeft,grayRight)
  denoise = 5
  noise=cv2.erode(disparity,np.ones((denoise,denoise)))
  noise=cv2.dilate(noise,np.ones((denoise,denoise)))
  disparity = cv2.medianBlur(disparity, ksize=5)
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
  try:
    depth[np.isinf(depth)] = np.max(depth[np.isfinite(depth)])
  except:
     pass
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
  # im = plt.imshow(depth)





  # print(np.shape(points3D))
  # opencv loads the image in BGR, so lets make it RGB
  colors = np.reshape(cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2RGB), (-1,3))
  # print(np.shape(colors))


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
  # print(projected_points)
  man_proj = KL @ test_pts
  man_proj = np.divide(man_proj,man_proj[-1,:])
  # print(man_proj)
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


  cv2.imshow('right',right)
  # cv2.waitKey()
  # im.set_data(depth)
  # plt.hist(depth)
  # plt.show()

  depth_thresh = 1.5 # Threshold for SAFE distance (in cm)

  # Mask to segment regions with depth less than threshold
  mask = cv2.inRange(depth,0.1,depth_thresh)
  # print(np.mean(depth))
  cv2.imshow('depth mask',mask)
  # cv2.waitKey()

  # Check if a significantly large obstacle is present and filter out smaller noisy regions
  if np.sum(mask)/255.0 > 0.02*mask.shape[0]*mask.shape[1]:
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
      if len(depth_mean_list) < 20:
        depth_mean_list.append(depth_mean[0,0])
      else:
         depth_mean_list.pop(0)
         depth_mean_list.append(depth_mean[0,0])
         
      # print(depth_mean[0])
      filtered_depth = np.mean(depth_mean_list)
      filtered_total_depth = scipy.signal.medfilt(total_depth_list)
      print(filtered_depth)
      if filtered_depth < 8:
        # Display warning text
        cv2.putText(fixedLeft, "WARNING !", (x+5,y-40), 1, 2, (0,0,255), 2, 2)
        cv2.putText(fixedLeft, "Object at", (x+5,y), 1, 2, (100,10,25), 2, 2)
        cv2.putText(fixedLeft, "%.2f cm"%filtered_depth, (x+5,y+40), 1, 2, (100,10,25), 2, 2)

  else:
    cv2.putText(fixedLeft, "SAFE!", (100,100),1,3,(0,255,0),2,3)
    if len(depth_mean_list) < 5:
        depth_mean_list.append(np.mean(depth))
    else:
        depth_mean_list.pop(0)
        depth_mean_list.append(np.mean(depth))
    filtered_depth = np.mean(depth_mean_list)

  print(filtered_depth)
  if len(total_depth_list) < 20:
    total_depth_list.append(np.mean(depth))
  else:
      total_depth_list.pop(0)
      total_depth_list.append(np.mean(depth))
         
  filtered_total_depth = scipy.signal.medfilt(total_depth_list)
  # print(filtered_total_depth[-1])
  cv2.imshow('output_canvas',fixedLeft)
  cv2.imshow('left', left)
  cv2.moveWindow("left", 10, 50) 
  cv2.moveWindow("right", 650, 50) 
  cv2.moveWindow("depth mask", 10, 500) 
  cv2.moveWindow("output_canvas", 650, 600) 
  cv2.waitKey(1)

  if first: 
    im = ax.imshow(depth)   
    plt.show()
    # mngr = plt.get_current_fig_manager()
    # geom = mngr.window.geometry()
    # x,y,dx,dy = geom.getRect()
    # mngr.window.setGeometry(10, 50, dx, dy)
    first = 0
  else:
    im.set_data(depth)
    fig.canvas.flush_events()