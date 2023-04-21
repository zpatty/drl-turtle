import cv2 as cv
import numpy as np

DIM=(1920, 1080)
KL=np.array([[936.3090354816636, 0.0, 1011.6603031360017], [0.0, 936.2302136924422, 542.9121240301195], [0.0, 0.0, 1.0]])
DL=np.array([[-0.0697808462865659], [-0.0031580486418187653], [0.0024539254744271204], [-0.0012931220378302362]])
KR=np.array([[935.5251123185054, 0.0, 990.8427164459354], [0.0, 936.1743386189506, 536.9829760404886], [0.0, 0.0, 1.0]])
DR=np.array([[-0.08446254512511926], [0.036252030276062254], [-0.03991208556882756], [0.01427235965589032]])
R=np.array([[-0.2169260764523986, 0.76437043407503, 0.6071910052585745], [0.13726143781936617, -0.591939119883656, 0.7942086476733448], [0.96648971802225, 0.25562847623712237, 0.023488445967295823]])
T=np.array([[1.6155752720185803], [2.1387015051205402], [-2.871774846881122]])

R1,R2,P1,P2,Q = cv.fisheye.stereoRectify(KL,DL,KR,DR,DIM,R,T,None)

#undistortion
L_undist_map=cv.fisheye.initUndistortRectifyMap(KL,DL,np.identity(3),KL,DIM,cv.CV_32FC1)
R_undist_map=cv.fisheye.initUndistortRectifyMap(KR,DR,np.identity(3),KR,DIM,cv.CV_32FC1)

#reprojectto3d
left1, left2 = cv.fisheye.initUndistortRectifyMap(KL, DL, R1, P1, DIM, cv.CV_32FC1)
right1, right2 = cv.fisheye.initUndistortRectifyMap(KR,DR, R2, P2, DIM, cv.CV_32FC1)

stereo=cv.StereoBM_create(16,15)

cap0 = cv.VideoCapture(0)
cap1 = cv.VideoCapture(1)
cv.namedWindow("main", cv.WINDOW_NORMAL)

while True:
    ret0, left = cap1.read()
    ret1, right = cap0.read()

    fixedLeft = cv.remap(left, L_undist_map[0], L_undist_map[1], cv.INTER_LINEAR)
    fixedRight = cv.remap(right, R_undist_map[0], R_undist_map[1], cv.INTER_LINEAR)

    # fixedLeft = cv.remap(left, left1, left2, cv.INTER_LINEAR)
    # fixedRight = cv.remap(right, right1, right2, cv.INTER_LINEAR)

    # grayLeft = cv.cvtColor(fixedLeft, cv.COLOR_BGR2GRAY)
    # grayRight = cv.cvtColor(fixedRight, cv.COLOR_BGR2GRAY)
    # depth = stereo.compute(grayLeft, grayRight)

    cv.imshow('left', fixedLeft)
    cv.imshow('right', fixedRight)
    # cv.imshow('depth', depth/2048)
    #cv.imshow('stereo', fixed)

    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cap0.release()
cap1.release()
cv.destroyAllWindows()