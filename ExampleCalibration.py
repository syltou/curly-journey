# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:51:15 2017

@author: sylvto
"""



import numpy as np
import cv2
from Geometry import undistortImage
from Calibration import getCalibrationImages, calibrateMonoCamera, calibrateStereoCamera, readCamerasFrames_20140523_MVcapture, predetectChessboard
from imgTools import show
import os

# path where the calibration imeages will be stored (must exist)
calib_path = u'.//Calib//'


# closing windows
cv2.destroyAllWindows()


def main():
    
    

    
#    ###------------------------------------------------------------------
#    ###-------------------------section starts here----------------------
#    ### Quick detection of the chessboard pattern and saving pictures
#    
#    predetectChessboard(calib_path)
#
#    ###-------------------------section ends here------------------------
#    ###------------------------------------------------------------------

    
#    ###------------------------------------------------------------------
#    ###-------------------------section starts here----------------------
#    ### Individual calibration of each camera
#    ### variable cam must be changed, as well as the names of the sample images 
#    ### paramaters (scaling_factor, refine) should be adapted to camera
#    ### - rgb cam 1920x1080 : scaling_factor=1, refine=True
#    ### - depth cam 160x120 : scaling_factor=3-5, refine=False
#    ### Correction of distortions is illustrated to ensure that results are ok
#    ### Results are stored in an NPZ archive file for reuse
#    
#
#    cams = ['left','middle','right','e40','e70']
#    scale = [1,1,1,4,3]
#    refine = [True,True,True,False,False]
#    threshold_average = [ 1.0, 1.0, 1.0, 0.25, 0.50]
#    threshold_individual = [ 1.0, 1.0, 1.0, 0.25, 0.50]
#    
#    for k in range(4,5):
#        
#        cam = cams[k]
#        archive = os.path.join(calib_path,'calib_'+cam+'.npz')
#        
#        # listing calibration images concerning selected camera
#        calibimages = getCalibrationImages(calib_path,cam)
#        # individual calibration from the calibration images
#        cameraMatrix, distCoeffs, imgpoints, calibrationimages = calibrateMonoCamera(calibimages, 
#                                                                                     reprojection_error_threshold_average = threshold_average[k],
#                                                                                     reprojection_error_threshold_individual = threshold_individual[k],
#                                                                                     scaling_factor = scale[k],
#                                                                                     refine_corners = refine[k],
#                                                                                     visualize = False)
#        print(cameraMatrix)
#        print(distCoeffs)
#    
#        # undistort 2 test images
#        img1 = readCamerasFrames_20140523_MVcapture(3500,4,cam)    
#        img1_undistort = undistortImage(img1, cameraMatrix, distCoeffs, crop=False)
#        show( img1, cam+' distorted1', height=600)
#        show( img1_undistort, cam+' undistorted1', height=600)
#        
#        img2 = readCamerasFrames_20140523_MVcapture(975,5,cam)
#        img2_undistort = undistortImage(img2, cameraMatrix, distCoeffs, crop=False)
#        show( img2, cam+' distorted2', height=600)
#        show( img2_undistort, cam+' undistorted2', height=600)
#        
#        # saving results
#        np.savez(archive, K=cameraMatrix, k=distCoeffs, imgpoints=imgpoints, calibrationimages=calibrationimages)
#
#    # display
#    cv2.waitKey(0)    
#    
#    ###-------------------------section ends here------------------------
#    ###------------------------------------------------------------------
    

    ###------------------------------------------------------------------
    ###-------------------------section starts here----------------------
    ### Stereo calibration
    
    cam1 = 'e40'
    cam2 = 'e70'
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = calibrateStereoCamera(cam1,cam2,calib_path)
    
    print(cameraMatrix1)
    print(cameraMatrix2)
    
    print('Rotation matrix')
    print(R)
    print('Translation vector')
    print(T)
    # Euler angles
    ret,_,_,_,_,_ = cv2.RQDecomp3x3(R)
    print('Rotation angles:',ret)
    
    
    img1 = readCamerasFrames_20140523_MVcapture(975,5,cam1)   
    img2 = readCamerasFrames_20140523_MVcapture(975,5,cam2)   

    
    show( img1, cam1+' original', height=500, position=(0,0))
    show( img2, cam2+' original', height=500, position=(int(500*img1.shape[1]/img1.shape[0]),0))
    
    
    (w1,h1)=img1.shape[1::-1]
    (w2,h2)=img2.shape[1::-1]
    
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify( cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (w1,h1), R, T, alpha=1, flags=cv2.CALIB_ZERO_DISPARITY)
    
    
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, img1.shape[1::-1], cv2.CV_32FC1 )
    img1_undist = cv2.remap(img1, mapx, mapy, cv2.INTER_LINEAR)
    
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, img2.shape[1::-1], cv2.CV_32FC1 )
    img2_undist = cv2.remap(img2, mapx, mapy, cv2.INTER_LINEAR)
    
    show( img1_undist, cam1+' rectified', height=500, position=(0,550))
    show( img2_undist, cam2+' rectified', height=500, position=(int(500*img1_undist.shape[1]/img1_undist.shape[0]),550))
    
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ###-------------------------section ends here------------------------
    ###------------------------------------------------------------------    
    
    
    
    
#    depthmap = readCamerasFrames_20140523_MVcapture(3500,4,'e70z')    
#    rgb = readCamerasFrames_20140523_MVcapture(3500,4,'left')  
#    new_depthmap = depthMapping( depthmap, cameraMatrix1, R, T, cameraMatrix2, rgb.shape[:2])
#    new_depthmap_undis = depthMapping( depthmap, cameraMatrix1, R, T, cameraMatrix2, rgb.shape[:2], True, distCoeffs2)
#    
#    show( depthmap, 'Original depthmap')
#    show( rgb, 'RGB image', height=800)
#    show( new_depthmap, 'Projected depthmap distorted', height=800)
#    show( new_depthmap_undis, 'Projected depthmap undistorted', height=800)
#    cv2.waitKey(0)
#    
#    
#    
#    mapx, mapy = cv2.initUndistortRectifyMap(calib['cameraMatrix_left'], calib['distCoeffs_left'], calib['R'], calib['cameraMatrix_right'], (1920,1080), cv2.CV_32FC1 )
#    new_img_left = cv2.remap(img_left, mapx, mapy, cv2.INTER_LINEAR)
#    
#    show(img_left)
#    show(img_right)
#    show(new_img_left)
#    cv2.waitKey(0)
    
    
    
#    depth_e70 = cv2.resize(depth_e70,(1920,1080))
    
    
#    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify( calib['cameraMatrix_left'], calib['distCoeffs_left'], calib['cameraMatrix_right'], calib['distCoeffs_right'], (1920,1080), calib['R'], calib['T'],flags=0)
#    
#    print(R1)
#    print(R2)
#    print(P1)
#    print(P2)
#    print(Q)
#    print(roi1)
#    print(roi2)
#    
#    mapx, mapy = cv2.initUndistortRectifyMap(calib['cameraMatrix_left'], calib['distCoeffs_left'], R2, P2, (1920,1080), cv2.CV_32FC1 )
#    new_img_left = cv2.remap(img_left, mapx, mapy, cv2.INTER_LINEAR)
#    x, y, w, h = roi1
#    new_img_left = new_img_left[y:y+h, x:x+w]
#    
#    mapx, mapy = cv2.initUndistortRectifyMap(calib['cameraMatrix_right'], calib['distCoeffs_right'], R1, P1, (1920,1080), cv2.CV_32FC1 )
#    new_img_right = cv2.remap(img_right, mapx, mapy, cv2.INTER_LINEAR)
#    x, y, w, h = roi2
#    new_img_right = new_img_right[y:y+h, x:x+w]
#    
#    show(img_left)
#    show(new_img_left)
#    show(img_right)
#    show(new_img_right)
#    cv2.waitKey(0)
    
    
#    K_e70 = np.matrix(calib['cameraMatrix_e70'])
#    R_e70 = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
#    T_e70 = np.array([[0],[0],[0]])
#    
#    K_left = np.matrix(calib['cameraMatrix_left'])
#    R_left = np.matrix(calib['R'])
#    T_left = np.matrix(calib['T'])
#    
#
#    
#    Rt_e70 = np.matrix(np.concatenate((R_e70,T_e70),1))
#    Rt_left = np.matrix(np.concatenate((R_left,T_left),1))
#    
#    print(Rt_e70)
#    print(Rt_left)
#    
#    
#    print(cv2.RQDecomp3x3(R_left))
#
#    depth_e70 = cv2.resize(depth_e70,(1920,1080))
#
#    newDepthMap = warpDepthMap(depth_e70,K_e70,Rt_e70,K_left,Rt_left,img_left.shape[1],img_right.shape[0])
#    
#    print(datetime.datetime.now())
#    
#    show(img_left)
#    show(depth_e70)
#    show(newDepthMap)
#    cv2.waitKey(0)
    


if __name__ == "__main__":
    main()
