import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import glob

rightimages = sorted(glob.glob('right/*.jpg'))
leftimages = sorted(glob.glob('left/*.jpg'))
print(rightimages)
# leftimages = ['L_1.png', 'L_2.png', 'L_0.png', 'L_3.png', 'L_4.png', 'L_5.png', 'L_6.png', 'L_7.png', 'L_8.png', 'L_9.png', 'L_11.png', 'L_12.png', 'L_10.png', 'L_13.png', 'L_14.png', 'L_15.png', 'L_16.png', 'L_17.png', 'L_18.png', 'L_19.png', 'L_21.png', 'L_22.png', 'L_20.png', 'L_23.png', 'L_24.png', 'L_25.png', 'L_26.png', 'L_27.png', 'L_28.png', 'L_29.png', 'L_30.png', 'L_33.png', 'L_34.png', 'L_35.png', 'L_36.png', 'L_37.png', 'L_38.png', 'L_39.png', 'L_41.png', 'L_42.png', 'L_40.png', 'L_43.png', 'L_44.png', 'L_45.png', 'L_46.png', 'L_47.png', 'L_48.png', 'L_49.png', 'L_50.png', 'L_51.png', 'L_52.png']
# rightimages = ['R_1.png', 'R_2.png', 'R_0.png', 'R_3.png', 'R_4.png', 'R_5.png', 'R_6.png', 'R_7.png', 'R_8.png', 'R_9.png', 'R_11.png', 'R_12.png', 'R_10.png', 'R_13.png', 'R_14.png', 'R_15.png', 'R_16.png', 'R_17.png', 'R_18.png', 'R_19.png','R_21.png', 'R_22.png', 'R_20.png', 'R_23.png', 'R_24.png', 'R_25.png', 'R_26.png', 'R_27.png', 'R_28.png', 'R_29.png', 'R_30.png', 'R_33.png', 'R_34.png', 'R_35.png', 'R_36.png', 'R_37.png', 'R_38.png', 'R_39.png', 'R_41.png', 'R_42.png', 'R_40.png', 'R_43.png', 'R_44.png', 'R_45.png', 'R_46.png', 'R_47.png', 'R_48.png', 'R_49.png', 'R_50.png', 'R_51.png',  'R_52.png']
H_list = []
calc_H = False

H = np.array([[ 9.92599166e-01, -2.58054123e-02, -2.40819858e+02],
				[ 2.82342128e-03,  9.54613600e-01,  3.46832646e+00],
				[-4.77848054e-05,  6.65765081e-07,  1.00000000e+00]])
H = np.array([[ 1.01874500e+00,  2.26559484e-02,  2.44314318e+02],
				 [-1.66611479e-03,  1.03952220e+00, -5.20914332e+00],
				 [ 5.86480321e-05,  1.02544032e-05,  1.00000000e+00]])

H = np.array([[ 1.81070092e-01, -1.60840998e-02,  3.59758126e+02],
				 [-1.62623906e-01,  6.96914977e-01,  5.89918198e+01],
				 [-7.46743680e-04, -1.08085711e-04,  1.00000000e+00]])

H=np.array([[0.825204996068902, 0.13058885087552038, 321.89335974619536], [-0.016272629698299568, 1.033175323488536, 8.402360154626619], [-0.00024551459450157574, 0.00020060583123329356, 1.0]])
# left_pool/frame117.jpg
# H=np.array([[1.1390113125348773, -0.06135107648330203, 311.27305378140414], [0.026647191179455534, 1.0758667074292285, -5.258403944429396], [0.00016043960097381573, -3.0616127937412854e-05, 1.0]])

for i in range(len(leftimages)):
	try:
		leftname=leftimages[i]
		rightname=rightimages[i]
		print(leftname)
		leftimg = cv2.imread(leftname)
		rightimg=cv2.imread(rightname)
		# img_ = cv2.imread('right/frame50.jpg')
		img2 = cv2.cvtColor(leftimg,cv2.COLOR_BGR2GRAY)
		# img = cv2.imread('left/frame50.jpg')
		img1 = cv2.cvtColor(rightimg,cv2.COLOR_BGR2GRAY)
		if calc_H:
			sift = cv2.xfeatures2d.SIFT_create()
			# find the keypoints and descriptors with SIFT
			kp1, des1 = sift.detectAndCompute(img1,None)
			kp2, des2 = sift.detectAndCompute(img2,None)

			bf = cv2.BFMatcher()
			matches = bf.knnMatch(des1,des2, k=2)

			# Apply ratio test
			good = []
			for m in matches:
				if m[0].distance < 0.5*m[1].distance:
					good.append(m)
			matches = np.asarray(good)

			if len(matches[:,0]) >= 4:
				src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
				dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
				H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
				print("H=np.array(" + str(H.tolist()) + ")")
				H_list.append(H)
			else:
				print("not enought keypoints")
				pass

			# dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))
			# plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
			# plt.show()
			# plt.figure()
			# dst[0:img.shape[0], 0:img.shape[1]] = img
			# # cv2.imwrite('output.jpg',dst)
			# plt.imshow(dst)
			# plt.show()

		# print(np.mean(H_list, axis=0))

			# Get the dimensions of the images
		h1, w1 = leftimg.shape[:2]
		h2, w2 = rightimg.shape[:2]
		# result = np.array(np.zeros((h1, 230 + w2,3)))
		result = np.concatenate((leftimg[:,0:230], rightimg), axis=1)
		# Get the canvas dimesions
		# pts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
		# dst = cv2.perspectiveTransform(pts, meanH)
		img2_warped = cv2.warpPerspective(rightimg, H, (w1 + w2, h1))

		# result = img2_warped
		# Place the first image on the canvas
		# img2_warped[0:h1, 0:w1] = leftimg

		# Simple blending technique
		# cv2.imwrite('new_' + rightname, img2_warped[:,255:860])
		
		# result = np.concatenate((leftimg[:,0:230], img2_warped[:,255:860]), axis=1)
		cv2.imshow('Result', result)

		key = cv2.waitKey(500)
		if key == 27:#if ESC is pressed, exit loop
	            cv2.destroyAllWindows()
	            break
	except:
		pass

cv2.destroyAllWindows()

