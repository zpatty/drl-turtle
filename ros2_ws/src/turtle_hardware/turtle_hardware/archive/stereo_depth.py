import cv2 as cv
import numpy as np

# DIM=(1920, 1080)
KL=np.array([[417.15751114066205, 0.0, 336.595336628034], [0.0, 416.8576501537559, 241.5489118345027], [0.0, 0.0, 1.0]])
DL=np.array([[-0.06815812211170555], [-0.016732544509364528], [0.029182156593969097], [-0.017701284426359723]])
DIM=(640, 480)
KR=np.array([[416.3903560278583, 0.0, 343.1831889045121], [0.0, 415.88140111385025, 241.99492603370734], [0.0, 0.0, 1.0]])
DR=np.array([[-0.06197454939758593], [-0.031440749408005376], [0.04248811930174599], [-0.02113466201121944]])
T = np.array([[65.1823933534524,  -4.73724842509345,   -20.8527190447127]])
R = np.array([[0.773850568208457,    0.0947576135973355,  0.626239804506858],
            [-0.142154340481768, 0.989504617178404,   0.0259375416108133],
            [-0.617209398474815,  -0.109094487706562,  0.779198916314955]])

R1,R2,P1,P2,Q = cv.fisheye.stereoRectify(KL,DL,KR,DR,DIM,R,T, cv.fisheye.CALIB_ZERO_DISPARITY)

#undistortion
L_undist_map=cv.fisheye.initUndistortRectifyMap(KL,DL,np.identity(3),KL,DIM,cv.CV_32FC1)
R_undist_map=cv.fisheye.initUndistortRectifyMap(KR,DR,np.identity(3),KR,DIM,cv.CV_32FC1)

#reprojectto3d
left1, left2 = cv.fisheye.initUndistortRectifyMap(KL, DL, R1, P1, DIM, cv.CV_32FC1)
right1, right2 = cv.fisheye.initUndistortRectifyMap(KR,DR, R2, P2, DIM, cv.CV_32FC1)

# stereo = cv.StereoSGBM_create(numDisparities=64, blockSize=29)
stereo = cv.StereoBM.create(numDisparities=64, blockSize=19)

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
    left = cv.imread("right/right 9.png")
    right = cv.imread("left/left 9.png")


    # just undistorted, no stereo
    # fixedLeft = cv.remap(left, L_undist_map[0], L_undist_map[1], cv.INTER_LINEAR)
    # fixedRight = cv.remap(right, R_undist_map[0], R_undist_map[1], cv.INTER_LINEAR)
    # cv.imshow("fixedLeft", fixedLeft)
    # cv.imshow("fixedRight", fixedRight)
    #stereo
    fixedLeft = cv.remap(left, left1, left2, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    fixedRight = cv.remap(right, right1, right2, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    # cv.imshow("fixedLeft", fixedLeft)
    # cv.imshow("fixedRight", fixedRight)

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