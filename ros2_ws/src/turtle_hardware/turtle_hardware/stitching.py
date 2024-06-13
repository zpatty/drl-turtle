# import the necessary packages
import numpy as np
# import imutils
import cv2
class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X and initialize the
        # cached homography matrix
        # self.isv3 = imutils.is_cv3()
        self.cachedH = None

    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        # unpack the images
        (imageB, imageA) = images
        # if the cached homography matrix is None, then we need to
        # apply keypoint matching to construct it
        if self.cachedH is None:
            # detect keypoints and extract
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)
            # match features between the two images
            M = self.matchKeypoints(kpsA, kpsB,
                featuresA, featuresB, ratio, reprojThresh)
            # if the match is None, then there aren't enough matched
            # keypoints to create a panorama
            if M is None:
                return None
            # cache the homography matrix
            self.cachedH = M[1]
        # apply a perspective transform to stitch the images together
        # using the cached homography matrix
        result = cv2.warpPerspective(imageA, self.cachedH,
            (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        # if self.isv3:
            # detect and extract features from the image
            #SIFT Algorithm
        descriptor = cv2.SIFT_create()
            #SURF Algorithm
            #descriptor = cv2.xfeatures2d.SURF_create()# 400 is hesian threshold, optimum values should be around 300-500
            #upright SURF: faster and can be used for panorama stiching i.e our case.
            #descriptor.upright = True
        print(descriptor.descriptorSize())
        (kps, features) = descriptor.detectAndCompute(image, None)
        print(len(kps),features.shape)

        # otherwise, we are using OpenCV 2.4.X
        # else:
        #     # detect keypoints in the image
        #     detector = cv2.FeatureDetector_create("SIFT")
        #     kps = detector.detect(gray)

        #     # extract features from the image
        #     extractor = cv2.DescriptorExtractor_create("SIFT")
        #     (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        #print("features",features)
        return (kps, features)