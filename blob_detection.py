#!/usr/bin/python

# Standard imports
import cv2
import numpy as np;

DIM=(1920, 1080)
KL=np.array([[914.6609693549937, 0.0, 996.710617938969], [0.0, 967.9244752752224, 531.9164424060089], [0.0, 0.0, 1.0]])
DL=np.array([[-0.1356783973167512], [0.15271796879021393], [-0.14927909026390898], [0.054553322922445247]])
KR=np.array([[894.3158759020713, 0.0, 1005.5147253984019], [0.0, 953.7162638446257, 550.0046766951555], [0.0, 0.0, 1.0]])
DR=np.array([[-0.03029069271100218], [-0.05098557630346465], [0.03042968864943995], [-0.007140226075471247]])
R=np.array([[0.8778242267055131, 0.03825565357540778, -0.4774527536609107], [-0.017035265337028843, 0.9986682915118547, 0.04869746670711228], [0.47867987919251936, -0.03461428171017962, 0.8773069159410083]])
T=np.array([[-3.0558948932592864], [0.09397400596710861], [-0.8536105947709979]])

R1,R2,P1,P2,Q = cv2.fisheye.stereoRectify(KL,DL,KR,DR,DIM,R,T, cv2.fisheye.CALIB_ZERO_DISPARITY)

L_undist_map=cv2.fisheye.initUndistortRectifyMap(KL,DL,np.identity(3),KL,DIM,cv2.CV_32FC1)
R_undist_map=cv2.fisheye.initUndistortRectifyMap(KR,DR,np.identity(3),KR,DIM,cv2.CV_32FC1)

left1, left2 = cv2.fisheye.initUndistortRectifyMap(KL, DL, R1, P1, DIM, cv2.CV_32FC1)
right1, right2 = cv2.fisheye.initUndistortRectifyMap(KR,DR, R2, P2, DIM, cv2.CV_32FC1)

# Read images
left = cv2.imread("balls/L_0.png")
right = cv2.imread("balls/R_0.png")

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200


# Filter by Area.
params.filterByArea = True
params.minArea = 50
params.maxArea = 100000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.1
    
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.1

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)

lower_yellow = np.array([15,50,180])
upper_yellow = np.array([40,255,255])

while(True):

    fixedLeft = cv2.remap(left, left1, left2, cv2.INTER_LINEAR)
    fixedRight = cv2.remap(right, right1, right2, cv2.INTER_LINEAR)
    
    # Converting from BGR to HSV color space
    hsv = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2HSV)
    
    # Compute mask
    premask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_not(premask)
    
    # Bitwise AND 
    # result = cv2.bitwise_and(im,im, mask= mask)

    # cv2.imshow('Mask',mask)
    # cv2.waitKey(0)
    # cv2.imshow('Masked Image',result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

    # Detect blobs.

    keypoints = detector.detect(mask)
    if not (len(keypoints) == 0):
        centroid = (keypoints[0].pt[0], keypoints[0].pt[1])
        print(centroid)
    # Determine largest blob (target) and centroid 
    #target_blob = keypoints
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)
    if cv2.waitKey(1) == 27:
        break
	
cv2.destroyAllWindows()

