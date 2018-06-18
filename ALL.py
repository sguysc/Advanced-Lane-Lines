# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 22:24:30 2018

@author: guyx64
"""
import glob
#import matplotlib.image as mpimg
import numpy as np
import cv2
#import matplotlib.pyplot as plt
from collections import deque

#%% Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
class LaneDetector:
    # load calibration images
    nx = 9 # the number of inside corners in x
    ny = 6 # the number of inside corners in y
    save_undistoted = True #for debug
    mtx = []
    dist = []
    Mpersp = []
    
    all_y = []
    tl_x = []
    tl_y = []
    tr_x = []
    tr_y = []
    left_fit = []
    right_fit = []
    left = deque()
    right = deque()
    left_fitx = []
    right_fitx = []
    Mpersp = []
    left_curverad = 0
    right_curverad = 0

    # calibration process based on calibration image Chessboard 
    def calibrate_camera(self, calib_images_path):
        images_path = glob.glob(calib_images_path)
        # store object points and image points
        objpoints = []
        imgpoints = []
        
        obj = np.zeros((self.nx*self.ny,3), np.float32)
        for i in range(self.nx):
            for j in range(self.ny):
                obj[i + j*self.nx, 0] = i
                obj[i + j*self.nx, 1] = j
                
        obj[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)
        
        for fname in images_path:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            if ret:
                objpoints.append(obj)
                imgpoints.append(corners)
#                img = cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
#                cv2.imshow('Chessboard',img)
            
        # Returns camera calibration
        ret, self.mtx, self.dist, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    def mask_car_and_sky(self, img, roi_corners):
#        shape = img.shape
#        vertices = np.array([[(0,0),(shape[1],0),(shape[1],0),(6*shape[1]/7,shape[0]),
#                      (shape[1]/7,shape[0]), (0,0)]],dtype=np.int32)
#
#        mask = np.zeros_like(img)   
#    
#        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
#        if len(img.shape) > 2:
#            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
#            ignore_mask_color = (255,) * channel_count
#        else:
#            ignore_mask_color = 255
#            
#        #filling pixels inside the polygon defined by "vertices" with the fill color    
#        cv2.fillPoly(mask, vertices, ignore_mask_color)
#        
#        #returning the image only where mask pixels are nonzero
#        masked_image = cv2.bitwise_and(img, mask)
#        return masked_image
        res=img.copy()
        #remove hood
        res[-int(res.shape[0]*0.1):-1,:,:]=0
        #remove sky
        res[0:int(roi_corners[1][1]),:,:]=0
        return res
    
    def warp(self, undist, roi_corners):
        new_top_left=np.array([roi_corners[0,0],0])
        new_top_right=np.array([roi_corners[3,0],0])
        offset=[150,0]
        
        img_size = (undist.shape[1], undist.shape[0])
        src = np.float32([roi_corners[0],roi_corners[1],roi_corners[2],roi_corners[3]])
        dst = np.float32([roi_corners[0]+offset,new_top_left+offset,new_top_right-offset ,roi_corners[3]-offset])    
        self.Mpersp = cv2.getPerspectiveTransform(src, dst)

        warped = cv2.warpPerspective(img, self.Mpersp, img_size , flags=cv2.INTER_LINEAR)    
        return warped
#        src = np.float32(roi_corners)
#    
#        offset=100
#        warped_size=(300+2*offset,300)
#        dst = np.float32([[offset, warped_size[1]],
#                          [offset, 0],
#                          [offset+warped_size[1], 0],
#                          [offset+warped_size[1], warped_size[1]]])
#    
#        self.Mpersp = cv2.getPerspectiveTransform(src, dst)
#        warped_img = cv2.warpPerspective(undist, self.Mpersp, dsize=warped_size)
#        return warped_img
    
    def mask_lane_lines(self, img, s_thresh=(120, 255), sx_thresh=(20, 255),l_thresh=(40,255)):
        img = np.copy(img)
    
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        #h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Sobel x
        # sobelx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255))
        # l_channel_col=np.dstack((l_channel,l_channel, l_channel))
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        # Threshold saturation channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
        # Threshold lightness
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
        
        channels = 255*np.dstack(( l_binary, sxbinary, s_binary)).astype('uint8')        
        binary = np.zeros_like(sxbinary)
        binary[((l_binary == 1) & (s_binary == 1) | (sxbinary==1))] = 1
#        binary = 255*np.dstack((binary,binary,binary)).astype('uint8')            
        return  255*binary.astype('uint8') 
#        img = np.copy(img)
#    
#        #Blur
#        kernel = np.ones((5,5),np.float32)/25
#        img = cv2.filter2D(img,-1,kernel)
#    
#        #YUV for histogram equalization
#        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
#        img_wht = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
#    
#        #Compute white mask
#        img_wht=img_wht[:,:,1]
#        img_wht[img_wht<250]=0
#        mask_wht = cv2.inRange(img_wht, 250, 255)
#    
#        yuv[:,:,0:1]=0
#    
#        #Yellow mask
#        kernel = np.ones((5,5),np.float32)/25
#        dst = cv2.filter2D(yuv,-1,kernel)
#        sobelx = np.absolute(cv2.Sobel(yuv[:,:,2], cv2.CV_64F, 1, 0, ksize=5))
#        sobelx[sobelx<200]=0
#        sobelx[sobelx>=200]=255
#    
#        #Merge mask results
#        mask = mask_wht + sobelx
#        return mask
    
    
    def find_line_points(self, img):
        points=([],[])
        shape=img.shape
        img=img.copy()
        #Prepare images for visualization
        red=np.zeros_like(img)
        green=np.zeros_like(img)
        blue=np.zeros_like(img)
    
        #Set center to width/2
        center=int(shape[1]/2)
    
        #For each row starting from bottom
        for yy,line in list(enumerate(img))[::-1]:
            x_val_hist=[]
            counter=0
            for x in line:
                if x>0:
                    x_val_hist.append(counter)
                counter=counter+1
            if len(x_val_hist)>0:
                cv2.circle(green,(int(center),yy),1,(255,255,255))
    
                #Split to left/right line
                left=[(x,yy) for x in x_val_hist if x<center]
                right=[(x,yy) for x in x_val_hist if x>=center]
    
                if len(left)>0:
                    #Compute average
                    l=np.mean(np.array(left),axis=0)
                    l=(l[0],l[1])
                    center=l[0]+int(shape[1]*0.2)
                    cv2.circle(red,(int(l[0]),int(l[1])),1,(255,255,255))
                    #Add to points
                    points[0].append(l)
                if len(right)>0:
                    #Compute average
                    r=np.mean(np.array(right),axis=0)
                    r=(r[0],r[1])
                    cv2.circle(blue,(int(r[0]),int(r[1])),1,(255,255,255))
                    #Add to points
                    points[1].append(r)
        if True: #for debug
            img=cv2.resize(np.dstack((blue,green,red)),(shape[1],int(shape[0])),fx=0,fy=0)
            cv2.imshow('lines',img)
            cv2.waitKey(2250)
            
        return points
    
    
    def fitAndShow(self, warped, undist, points):
        N_frames=25
        if len(points[0]) >= 5 and len(points[1])>=5:
            leftx,lefty= zip(*points[0])
            rightx,righty= zip(*points[1])
    
            self.all_y=np.array(list(range(warped.shape[0])))
    
            self.tl_x=list(leftx)
            self.tl_y=list(lefty)
            self.tr_x=list(rightx)
            self.tr_y=list(righty)
    
            #convert to numpy
            self.tl_x=np.array(self.tl_x)
            self.tl_y=np.array(self.tl_y)
            self.tr_x=np.array(self.tr_x)
            self.tr_y=np.array(self.tr_y)
    
            # Fit a second order polynomial to each fake lane line
            self.left_fit = np.array(np.polyfit(self.tl_y, self.tl_x, 2))
            self.right_fit = np.array(np.polyfit(self.tr_y, self.tr_x, 2))
    
            self.left.append(self.left_fit)
            if len(self.left)>=N_frames:
                self.left.popleft()
            self.right.append(self.right_fit)
            if len(self.right)>=N_frames:
                self.right.popleft()
    
    
            self.left_fit=np.array([0,0,0])
            for v in self.left:
                self.left_fit = self.left_fit + v
            self.left_fit=self.left_fit/len(self.left)
            self.right_fit=np.array([0,0,0])
            for v in self.right:
                self.right_fit = self.right_fit + v
            self.right_fit=self.right_fit/len(self.right)
    
            self.left_fitx = np.array(self.left_fit[0]*self.all_y**2 + self.left_fit[1]*self.all_y + self.left_fit[2])
            self.right_fitx = np.array(self.right_fit[0]*self.all_y**2 + self.right_fit[1]*self.all_y + self.right_fit[2])
    
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.all_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.all_y])))])
        pts = np.hstack((pts_left, pts_right))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv=np.linalg.inv(self.Mpersp)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.2, 0)
    
        unwarped = 255*cv2.warpPerspective(warped, Minv, (undist.shape[1], undist.shape[0])).astype(np.uint8)
        unwarped=np.dstack((np.zeros_like(unwarped),np.zeros_like(unwarped),unwarped))
    
        line_image = cv2.addWeighted(result, 1, unwarped, 1, 0)
        return result,line_image
    
    def computeAndShow(self, img, warped):
            # Define y-value where we want radius of curvature
            y_eval = np.max(self.all_y)
    
            # Define conversions in x and y from pixels space to meters
            ym_per_pix = 60/720 # meters per pixel in y dimension
            xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(np.array(self.tl_y)*ym_per_pix, 
                                     np.array(self.tl_x)*xm_per_pix, 2)
            right_fit_cr = np.polyfit(np.array(self.tr_y)*ym_per_pix, 
                                      np.array(self.tr_x)*xm_per_pix, 2)
    
            # Calculate the new radii of curvature
            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
            right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
            #Distance from center
            img_center=warped.shape[1]/2
            lane_center=(self.left_fitx[-1]+self.right_fitx[-1])*0.5
            diff=lane_center-img_center
            diffm=diff*xm_per_pix
    
            img=cv2.putText(img,'Curvature left: %.1f m'%(left_curverad),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
            img=cv2.putText(img,'Curvature right: %.1f m'%(right_curverad),(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
            img=cv2.putText(img,'Dist from center: %.2f m'%(diffm),(50,150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    
    
    def process_image(self, img):
        # initialize for each picture
        self.left.clear()
        self.right.clear()
        
        #undistort after calibration
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        
#        roi_corners=[[0.16*undist.shape[1],undist.shape[0]],
#                     [0.45*undist.shape[1],0.63*undist.shape[0]],
#                     [0.55*undist.shape[1],0.63*undist.shape[0]],
#                     [0.84*undist.shape[1],undist.shape[0]]]
        roi_corners = np.float32([[160, 719], [590,440], [740, 440], [1170, 719]])
    
        show_roi=True
        if show_roi:
            src = np.float32(roi_corners)
            pts = np.array(src, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(undist,[pts],True,(0,0,255))
            cv2.imshow('roi',undist)
#            cv2.waitKey(2250)
    
        #remove unwanted image data (sky, car)
        undist_masked = self.mask_car_and_sky(undist, roi_corners)
        show_mask=True
        if show_mask:
            cv2.imshow('undist_masked',undist_masked)
#            cv2.waitKey(2250)
    
        #Save before and after undistortion one time only
        if self.save_undistoted:
            before_after=np.concatenate((img, undist), axis=1)
            before_after=cv2.putText(before_after,'Distorted example pic', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            before_after=cv2.putText(before_after,'Undistorted example pic', (50+img.shape[1],50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
            cv2.imwrite('output_images\\distortion_fig.jpg',before_after)
            self.save_undistoted=False
    
        warped = self.warp(undist_masked, roi_corners)
        show_warped=True
        if show_warped:
#            before_after=np.concatenate((undist, warped), axis=1)
#            before_after=cv2.putText(before_after,'Undistorted example pic', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
#            before_after=cv2.putText(before_after,'warped example pic', (50+img.shape[1],50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('warped_masked',warped)
#            cv2.waitKey(4250)
            
        # cv2.imshow('transformed',warped)
        warped_mask = self.mask_lane_lines(warped)
        show_mask_line=True
        if show_mask_line:
            before_after=np.concatenate((np.dstack((warped_mask,warped_mask,warped_mask)).astype('uint8'), warped), axis=1)
            before_after=cv2.putText(before_after,'warped original example pic', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            before_after=cv2.putText(before_after,'warped mask example pic', (50+warped_mask.shape[1],50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('mask_line', before_after) #warped_mask)
#            cv2.waitKey(4250)
        
        # cv2.imshow('binary',warped)
        points = self.find_line_points(warped_mask)
        # cv2.imshow('warped',warped)
    
        if len(points[0]) == 0 or len(points[1])==0:
            return img
    
        result, line_image = self.fitAndShow(warped_mask, undist, points)
        self.computeAndShow(result, warped_mask)
    
        return result

#%%
if __name__ == '__main__':
    laneDet = LaneDetector()
    
    laneDet.calibrate_camera('camera_cal\\*.jpg')
    
    for fimg in glob.glob('test_images\\*.jpg'):
        fname = fimg.split('\\')[-1]
        print("processing file " + fname)
        img=cv2.imread(fimg)
        #Do processing
        res = laneDet.process_image(img)
        #Write result
        cv2.imwrite('test_images\\p_' + fname, res)
        #show results
#        cv2.imshow('Result', res)
        cv2.waitKey(250)
        break
    
    
    #Process all videos
#    for fvideo in glob.glob('*.mp4'):
#        process_video(fvideo)
    
    print('done')
    
    
