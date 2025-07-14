import numpy as np
import cv2 
import json
import os

class StereoProcessor():
    def __init__(self):
        cv_params, __ = self.parse_cv_params()

        DIM=(640, 480)
        # KL=np.array([[418.862685263043, 0.0, 344.7127322430985], [0.0, 418.25718294545146, 244.86612342606873], [0.0, 0.0, 1.0]])
        # DL=np.array([[-0.10698601862156384], [0.1536626773154175], [-0.25203748540815346], [0.134699123195767]])
        # KR=np.array([[412.8730413534633, 0.0, 334.94298327120686], [0.0, 413.2522575868915, 245.1860564579], [0.0, 0.0, 1.0]])
        # DR=np.array([[0.003736892852052395], [-0.331577509789992], [0.5990981643072193], [-0.3837158104256219]])
        # R=np.array([[0.8484938703183661, -0.053646440984050164, -0.5264790702410727], [0.060883267910572095, 0.9981384545889453, -0.00358513030742383], [0.5256913350253065, -0.029011805192654422, 0.8501805310866477]])
        # T=np.array([[-2.178632057371688], [-0.03710693058315735], [-0.6477466090945703]])*25.4

        # KL=np.array([[461.2096791449331, 0.0, 294.1931266259774], [0.0, 454.96037055881453, 263.01440757343636], [0.0, 0.0, 1.0]])
        # DL=np.array([[-0.012432450773748072], [-0.19716381408758557], [0.3098456749154318], [-0.13629078768961578]])
        # KR=np.array([[458.4349428541636, 0.0, 344.9511721760442], [0.0, 458.4949519104176, 255.16175894368138], [0.0, 0.0, 1.0]])
        # DR=np.array([[-0.01436160384200864], [-0.43814038651754705], [0.9634406292419895], [-0.7431950832159185]])
        # R=np.array([[0.8094197385611023, 0.020731284334064997, -0.5868644653389202], [-0.01700321757440054, 0.9997850140562221, 0.0118666027453989], [0.586984307643564, 0.00037352169905266984, 0.8095982232328243]])
        # T=np.array([[-2.1605934937846447], [0.12095314113784378], [-0.8424493082850986]])*25.4

        ############# May 16th OG Intrinsics used for all trials ##########################33
        # KL=np.array([[567.5108694496967, 0.0, 296.65113980779455], [0.0, 555.4066963178193, 209.8357339941157], [0.0, 0.0, 1.0]])
        # DL=np.array([[2.3304306359821676], [-22.72098020337239], [96.18495273925535], [-142.38468386964402]])
        # KR=np.array([[654.417602210813, 0.0, 236.58996237455327], [0.0, 625.1929731011552, 223.78494035869645], [0.0, 0.0, 1.0]])
        # DR=np.array([[-1.4455817082295017], [22.451030896752517], [-138.62442550617396], [291.546248181601]])
        # R=np.array([[0.9350531687261983, 0.001570069165988603, -0.3545040289445381], [-0.007541738734551825, 0.9998519807567543, -0.01546411179650275], [0.3544272758013351, 0.01713334350350178, 0.934926603915214]])
        # T=np.array([[-2.1428084390335562], [-0.02825237240566071], [-0.36330855915337945]])*25.4

        KL=np.array([[708.3477312219868, 0.0, 260.69187590557686], [0.0, 675.3059166594338, 301.31936629865646], [0.0, 0.0, 1.0]])
        DL=np.array([[-0.39383047117877457], [6.721465255404687], [-35.99917141986595], [61.49579122578909]])
        KR=np.array([[667.0400978057647, 0.0, 334.8109094526051], [0.0, 644.922628956739, 364.07228200370565], [0.0, 0.0, 1.0]])
        DR=np.array([[0.8809516193294453], [-6.609640306922403], [21.549513701823056], [-24.149385093847197]])
        R=np.array([[0.8721459388442752, 0.02130940474841954, -0.4887815162490354], [-0.06589130347707366, 0.9950649291385254, -0.07418977641584823], [0.4847884048567273, 0.09691076342599622, 0.8692459412897253]])
        T=np.array([[-2.085136618149882], [0.1939622251215522], [-0.9258137973647751]])*25.4

        #### OFFICE ####
        # KL=np.array([[423.1463253724279, 0.0, 339.7655580118522], [0.0, 422.4740092463073, 251.6373695701372], [0.0, 0.0, 1.0]])
        # DL=np.array([[-0.1609640159909826], [0.45720111661664947], [-1.2217642208776298], [1.169402105578575]])
        # KR=np.array([[405.75878487386933, 0.0, 315.45962600644606], [0.0, 409.5904306872554, 266.9218843465606], [0.0, 0.0, 1.0]])
        # DR=np.array([[-0.07903979401711092], [0.030562251976355453], [-0.016231525162586774], [-0.006748441817726353]])
        # R=np.array([[0.899732710386163, -0.0011003927241870328, -0.4364399603576895], [-0.01428880227470787, 0.9993864787298194, -0.031976495454729734], [0.43620738186179675, 0.035006503222467164, 0.8991649819368035]])
        # T=np.array([[-2.2310954859764314], [0.13544012980046657], [-0.9152593687088619]])*25.4
        R1,R2,P1,P2,self.Q = cv2.fisheye.stereoRectify(KL,DL,KR,DR,DIM,R,T, cv2.fisheye.CALIB_ZERO_DISPARITY)
        print(self.Q)
        self.L_undist_map=cv2.fisheye.initUndistortRectifyMap(KL,DL,np.identity(3),KL,DIM,cv2.CV_32FC1)
        self.R_undist_map=cv2.fisheye.initUndistortRectifyMap(KR,DR,np.identity(3),KR,DIM,cv2.CV_32FC1)
        self.left1, self.left2 = cv2.fisheye.initUndistortRectifyMap(KL, DL, R1, P1, DIM, cv2.CV_32FC1)
        self.right1, self.right2 = cv2.fisheye.initUndistortRectifyMap(KR,DR, R2, P2, DIM, cv2.CV_32FC1)
        self.stereo = cv2.StereoBM.create(numDisparities=cv_params["numDisparities"], blockSize=cv_params["blockSize"])
        self.stereo.setMinDisparity(cv_params["MinDisparity"])
        self.stereo.setTextureThreshold(cv_params["TextureThreshold"])

        # post filtering parameters: prevent false matches, help filter at boundaries
        self.stereo.setSpeckleRange(cv_params["SpeckleRange"])
        self.stereo.setSpeckleWindowSize(cv_params["SpeckleWindowSize"])
        self.stereo.setUniquenessRatio(cv_params["UniquenessRatio"])

        self.stereo.setDisp12MaxDiff(cv_params["Disp12MaxDiff"])
        self.depth_mean_list = [10000000]
        self.num_samples = 10

    def stereo_params_update(self):
        if self._last_update != os.stat('cv_config.json').st_mtime:
            cv_params, __ = self.parse_cv_params()
            self.stereo.setNumDisparities(cv_params["numDisparities"])
            self.stereo.setBlockSize(cv_params["blockSize"])
            self.stereo.setMinDisparity(cv_params["MinDisparity"])
            self.stereo.setTextureThreshold(cv_params["TextureThreshold"])

            # post filtering parameters: prevent false matches, help filter at boundaries
            self.stereo.setSpeckleRange(cv_params["SpeckleRange"])
            self.stereo.setSpeckleWindowSize(cv_params["SpeckleWindowSize"])
            self.stereo.setUniquenessRatio(cv_params["UniquenessRatio"])

            self.stereo.setDisp12MaxDiff(cv_params["Disp12MaxDiff"])
    def parse_cv_params(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'cv_config.json')
        with open(config_path) as config:
            param = json.load(config)
            self._last_update = os.fstat(config.fileno()).st_mtime
        print(f"[MESSAGE] Config: {param}\n")    
        # Serializing json
        config_params = json.dumps(param, indent=14)
        return param, config_params
    def update(self, left, right):
        fixedLeft = cv2.remap(left, self.left1, self.left2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        fixedRight = cv2.remap(right, self.right1, self.right2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo.compute(grayLeft,grayRight)
        denoise = 5
        noise=cv2.erode(disparity,np.ones((denoise,denoise)))
        noise=cv2.dilate(noise,np.ones((denoise,denoise)))
        disparity = cv2.medianBlur(noise, ksize=5)
        # norm_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        invalid_pixels = disparity < 0.0001
        disparity[invalid_pixels] = 0
        norm_disparity = np.array((disparity/16.0 - self.stereo.getMinDisparity())/self.stereo.getNumDisparities(), dtype='f')
        self.norm_disparity = norm_disparity
        points3D = cv2.reprojectImageTo3D(np.array(disparity/16.0/1000.0, dtype='f'), self.Q, handleMissingValues=True)
        depth = self.Q[2,3]/self.Q[3,2]/np.array(disparity/16.0, dtype='f')/1000
        x_bounds = [158,613]
        y_bounds = [30,450]
        depth_window = depth[30:450,158:613]
        finite_depth = depth_window[np.isfinite(depth_window)]
        stop_mean = np.median(finite_depth)
        h_thresh = 80
        w_thresh = 100
        depth[np.isinf(depth)] = np.median(finite_depth)
        # print(np.mean(depth))
        
        depth_thresh = 1.5 # Threshold for SAFE distance (in cm)

        # Mask to segment regions with depth less than threshold
        mask = cv2.inRange(depth,0.1,depth_thresh)
        print(self.depth_mean_list)
        # Check if a significantly large obstacle is present and filter out smaller noisy regions
        if np.sum(mask)/255.0 > 0.01*mask.shape[0]*mask.shape[1]:
            
            # Contour detection 
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Check if detected contour is significantly large (to avoid multiple tiny regions)
            if cv2.contourArea(cnts[0]) > 0.03*mask.shape[0]*mask.shape[1]:
                
                x,y,w,h = cv2.boundingRect(cnts[0])
                x_center = int(x + w/2)
                y_center = int(y + h/2)
                # finding average depth of region represented by the largest contour 
                mask2 = np.zeros_like(mask)
                # cv2.imshow("depth", mask)
                # cv2.waitKey(1)
                # cv2.drawContours(mask2, cnts, 0, (255), -1)
                # cv2.drawContours(norm_disparity, cnts, 0, (255), -1)
                # Calculating the average depth of the object closer than the safe distance
                depth_mean, _ = cv2.meanStdDev(depth, mask=mask)
                
                # Display warning text
                # cv2.putText(norm_disparity, "WARNING !", (x+5,y-40), 1, 2, (0,0,255), 2, 2)
                # cv2.putText(norm_disparity, "Object at", (x+5,y), 1, 2, (100,10,25), 2, 2)
                # cv2.putText(norm_disparity, "%.2f m"%depth_mean, (x+5,y+40), 1, 2, (100,10,25), 2, 2)
                
                # STEER AWAY HERE
                # print(depth_thresh/depth_mean[0,0])
                # print(mask2)
                # - 1.0 + 2.0 * 1 / (1 + np.exp((depth_mean[0,0] - depth_thresh)/depth_thresh))
                depth_mean, _ = cv2.meanStdDev(depth, mask=mask2)
                if len(self.depth_mean_list) < self.num_samples:
                    self.depth_mean_list.append(depth_mean[0,0])
                else:
                    self.depth_mean_list.pop(0)
                    self.depth_mean_list.append(depth_mean[0,0])
                    
                print(self.depth_mean_list)
                filtered_depth = np.mean(self.depth_mean_list)
                print(filtered_depth)
                return filtered_depth, x, y, norm_disparity
            else:
                cv2.putText(norm_disparity, "SAFE!", (100,100),1,3,(0,255,0),2,3)
                if len(self.depth_mean_list) < self.num_samples:
                    self.depth_mean_list.append(np.mean(depth))
                else:
                    self.depth_mean_list.pop(0)
                    self.depth_mean_list.append(np.mean(depth))
                filtered_depth = np.mean(self.depth_mean_list)
                print(filtered_depth)
                return filtered_depth, None, None, norm_disparity
        else:
            if len(self.depth_mean_list) < self.num_samples:
                    self.depth_mean_list.append(np.mean(depth))
            else:
                self.depth_mean_list.pop(0)
                self.depth_mean_list.append(np.mean(depth))
            filtered_depth = np.mean(self.depth_mean_list)
            return filtered_depth, None, None, norm_disparity
