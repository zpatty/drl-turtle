#!/usr/bin/python

# Standard imports
import cv2
import numpy as np
import json 
def parse_cv_params():
    with open('cv_config.json') as config:
        param = json.load(config)
    print(f"[MESSAGE] Config: {param}\n")    
    # Serializing json
    config_params = json.dumps(param, indent=14)
    return param, config_params
cv_params, __ = parse_cv_params()

DIM=(640, 480)
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
# left = cv2.imread("balls/L_0.png")
# right = cv2.imread("balls/R_0.png")

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = cv_params["minThreshold"]
params.maxThreshold = cv_params["maxThreshold"]


# Filter by Area.
params.filterByArea = True
params.minArea = cv_params["minArea"]
params.maxArea = cv_params["maxArea"]

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = cv_params["minCircularity"]

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = cv_params["minConvexity"]
    
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = cv_params["minInertiaRatio"]


# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)

lower_yellow = np.array(cv_params["lower_yellow"])
upper_yellow = np.array(cv_params["upper_yellow"])
cap0 = cv2.VideoCapture(4)
# cap0.set(3, 1920)
# cap0.set(4, 1080)
# cap1 = cv2.VideoCapture(2)

def fixHSVRange(val):
    # Normal H,S,V: (0-360,0-100%,0-100%)
    # OpenCV H,S,V: (0-180,0-255 ,0-255)
    return (180 * val[0] / 360, 255 * val[1] / 100, 255 * val[2] / 100)

while(True):

    # ret0, left = cap1.read()
    ret1, right = cap0.read()
    denoise = 15
    # noise=cv2.erode(right,np.ones((denoise,denoise)))
    # noise=cv2.dilate(noise,np.ones((denoise,denoise)))
    # right=cv2.medianBlur(noise,7)
    # fixedLeft = cv2.remap(left, left1, left2, cv2.INTER_LINEAR)
    # fixedRight = cv2.remap(right, right1, right2, cv2.INTER_LINEAR)
    blur = cv2.GaussianBlur(right, (5,5), 1)
    # Converting from BGR to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lab = cv2.cvtColor(right, cv2.COLOR_BGR2LAB)
    # cv2.imshow('Mask',lab[:,:,2])
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    # Compute mask
    # lb_y = np.array([11, 0, 153])
    # ub_y = np.array([52, 255, 255])
    # bin_y = cv2.inRange(hsv, lb_y, ub_y)
    bin_y = cv2.inRange(hsv, fixHSVRange(lower_yellow), fixHSVRange(upper_yellow))
    open_kern = np.ones((10,10), dtype=np.uint8)
    bin_y = cv2.morphologyEx(bin_y, cv2.MORPH_OPEN, open_kern, iterations=2)

    rip_y = right.copy()
    rip_y[bin_y==0] = 0
    mark_y = cv2.addWeighted(right, .4, rip_y, .6, 1)
    cv2.imshow("marked yellow", bin_y)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # premask = cv2.inRange(hsv, fixHSVRange(lower_yellow), fixHSVRange(upper_yellow))
    # mask = cv2.bitwise_not(premask)
    
    # # Bitwise AND 
    # # result = cv2.bitwise_and(im,im, mask= mask)
    
    # mask=cv2.cvtColor(premask,cv2.COLOR_GRAY2BGR)
    # noise=cv2.cvtColor(premask,cv2.COLOR_GRAY2BGR)
    # cleanMask=~noise
    # centerMask=cv2.cvtColor(cleanMask.copy(),cv2.COLOR_BGR2GRAY)

    # cv2.imshow('Mask',centerMask)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # cnts, _ = cv2.findContours(centerMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # # print(cnts)
    # # cnts.sort(key=lambda p: cv2.boundingRect(p)[2])

    # # Find and draw blocks
    # for cnt in cnts:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     if w<DIM[0]:
    #         cv2.rectangle(right, (x, y), (x+w, y+h), (0, 255, 0), 1)
    #         cv2.circle(right,(x+w//2,y+h//2),radius=5,thickness=5,color=(127,0,220))
    #         cv2.rectangle(centerMask, (x, y), (x+w, y+h), 127, 1)
    # centerMask=cv2.cvtColor(centerMask,cv2.COLOR_GRAY2BGR)
    # cv2.imshow('Mask',centerMask)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # cv2.waitKey(0)
    # cv2.imshow('Masked Image',result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

    # Detect blobs.

    # keypoints = detector.detect(centerMask)
    # if not (len(keypoints) == 0):
    #     centroid = (keypoints[0].pt[0], keypoints[0].pt[1])
    #     print(centroid)
    # # Determine largest blob (target) and centroid 
    # #target_blob = keypoints
    # # Draw detected blobs as red circles.
    # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # # the size of the circle corresponds to the size of blob

    #     im_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #     # Show blobs
    #     cv2.imshow("Keypoints", im_with_keypoints)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     turn_thresh = cv_params["turn_thresh"]
    #     dive_thresh = cv_params["dive_thresh"]
    #     if abs(centroid[0] - 1920/2) < turn_thresh and abs(centroid[1] - 1080/2) < dive_thresh:
    #         # output straight primitive
    #         print("go straight...\n")
    #     elif centroid[0] > 1920 - (1920/2 - turn_thresh):
    #         # turn right
    #         print("turn right...\n")
    #     elif centroid[0] < (1920/2 - turn_thresh):
    #         # turn left
    #         print("turn left...\n")
    #     elif centroid[1] > 1080 - (1080/2 - dive_thresh):
    #         # dive
    #         print("dive...\n")
    #     elif centroid[1] < (1080/2 - dive_thresh): 
    #         # surface
    #         print("surface...\n")
    #     else:
    #         # dwell
    #         print("dwell...\n")

	
cv2.destroyAllWindows()
