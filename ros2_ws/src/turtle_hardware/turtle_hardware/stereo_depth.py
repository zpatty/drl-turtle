import cv2 as cv
import numpy as np

DIM=(1920, 1080)
KL=np.array([[914.6609693549937, 0.0, 996.710617938969], [0.0, 967.9244752752224, 531.9164424060089], [0.0, 0.0, 1.0]])
DL=np.array([[-0.1356783973167512], [0.15271796879021393], [-0.14927909026390898], [0.054553322922445247]])
KR=np.array([[894.3158759020713, 0.0, 1005.5147253984019], [0.0, 953.7162638446257, 550.0046766951555], [0.0, 0.0, 1.0]])
DR=np.array([[-0.03029069271100218], [-0.05098557630346465], [0.03042968864943995], [-0.007140226075471247]])
R=np.array([[0.8778242267055131, 0.03825565357540778, -0.4774527536609107], [-0.017035265337028843, 0.9986682915118547, 0.04869746670711228], [0.47867987919251936, -0.03461428171017962, 0.8773069159410083]])
T=np.array([[-3.0558948932592864], [0.09397400596710861], [-0.8536105947709979]])

R1,R2,P1,P2,Q = cv.fisheye.stereoRectify(KL,DL,KR,DR,DIM,R,T, cv.fisheye.CALIB_ZERO_DISPARITY)

#undistortion
L_undist_map=cv.fisheye.initUndistortRectifyMap(KL,DL,np.identity(3),KL,DIM,cv.CV_32FC1)
R_undist_map=cv.fisheye.initUndistortRectifyMap(KR,DR,np.identity(3),KR,DIM,cv.CV_32FC1)

#reprojectto3d
left1, left2 = cv.fisheye.initUndistortRectifyMap(KL, DL, R1, P1, DIM, cv.CV_32FC1)
right1, right2 = cv.fisheye.initUndistortRectifyMap(KR,DR, R2, P2, DIM, cv.CV_32FC1)

stereo = cv.StereoSGBM_create(numDisparities=64, blockSize=29)
#prefiltering parameters: indended to normalize brightness & enhance texture
# stereo.setPreFilterSize(155)
# #^must be odd & btwn 5 & 255
# stereo.setPreFilterCap(7)
#^must be btwn 1 and 63
# stereo.setPreFilterType(2)

#stereo correspondence parameters: find matches between camera views
stereo.setMinDisparity(0)
# stereo.setTextureThreshold(2)

#post filtering parameters: prevent false matches, help filter at boundaries
stereo.setSpeckleRange(0)
stereo.setSpeckleWindowSize(5)
stereo.setUniquenessRatio(2)

stereo.setDisp12MaxDiff(2)

# cap0 = cv.VideoCapture(1)
# cap1 = cv.VideoCapture(2)
cv.namedWindow("main", cv.WINDOW_NORMAL)

while True:
    # ret0, left = cap1.read()
    # ret1, right = cap0.read()
    left = cv.imread("right/right 1.png")
    right = cv.imread("left/left 1.png")


    #just undistorted, no stereo
    # fixedLeft = cv.remap(left, L_undist_map[0], L_undist_map[1], cv.INTER_LINEAR)
    # fixedRight = cv.remap(right, R_undist_map[0], R_undist_map[1], cv.INTER_LINEAR)

    #stereo
    # fixedLeft = cv.remap(left, left1, left2, cv.INTER_LINEAR)
    # fixedRight = cv.remap(right, right1, right2, cv.INTER_LINEAR)

    grayLeft = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
    grayRight = cv.cvtColor(right, cv.COLOR_BGR2GRAY)
    disparity = stereo.compute(grayLeft,grayRight)
    norm_disparity = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    # local_max = disparity.max()
    # local_min = disparity.min()
    # disparity_visual = (disparity-local_min)*(1.0/(local_max-local_min))
    cv.imshow("depth", norm_disparity)

    # cv.imshow('left', grayLeft)
    # cv.imshow('right', grayRight)
    
    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

# cap0.release()
# cap1.release()
cv.destroyAllWindows()