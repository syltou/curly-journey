# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:51:15 2017

@author: sylvto
"""


import numpy as np
import cv2
import glob
from pprint import pprint
import os, sys
from imgTools import readPGM, show, warpDepthMap
import pickle
import datetime,time
from Geometry import distortPoints, reprojectToCalibrationImage




def readCamerasFrames_20140523_MVcapture(left_index,seq=5, name=None):
    ''' This function returns the images from the dataset "20140523_MVcapture"
        for a given index of the leftmost camera. Other indexes are deduced 
        from parameters below.
        Synchronization has been performed manually and is probably not that 
        accurate '''
    
    # path of the "20140523_MVcapture" dataset
    datapath = u'//ID20809271/LFdata/'
    
    
    if(seq!=4 and seq!=5):
        print('In grabImages(left_index,seq), seq variable indicates from which sequence (4 or 5) the frames should be grabbed. Only 4 or 5 are accpeted values')
        return -1
    
    else:
        
        if(seq==4):
            # SEQUENCE #4
            # color cams
            cam_left = datapath+'Left/SEQ4_LEFT_MVI_0393.MOV'
            cam_middle = datapath+'Middle/SEQ4_MIDDLE_MVI_0003.MOV'
            cam_right = datapath+'Right/SEQ4_RIGHT_MVI_0567.MOV'
            middle_offset = int(+22)
            right_offset = int(-30)
            # depth cams 
            cam70_b = datapath+'E70/SEQ4_E70_FZ_2014-05-23_11250601_b.pgm'
            cam70_z = datapath+'E70/SEQ4_E70_FZ_2014-05-23_11250601_z.pgm'
            cam40_b = datapath+'E40/SEQ4_E40_FZ_2014-05-23_11250701_b.pgm'
            cam40_z = datapath+'E40/SEQ4_E40_FZ_2014-05-23_11250701_z.pgm'
            cam70_offset = -178
            cam70_factor = 0.963
            cam40_offset = -212
            cam40_factor = 0.963
            max_depth_e40 = 9643
            max_depth_e70 = 10207
            
            
        if(seq==5):
            # SEQUENCE #5
            # color cams
            cam_left = datapath+'Left/SEQ5_LEFT_MVI_0394.MOV'
            cam_middle = datapath+'Middle/SEQ5_MIDDLE_MVI_0004.MOV'
            cam_right = datapath+'Right/SEQ5_RIGHT_MVI_0568.MOV'
            middle_offset = int(+27)
            right_offset = int(+62)
            # depth cams 
            cam70_b = datapath+'E70/SEQ5_E70_FZ_2014-05-23_11333301_b.pgm'
            cam70_z = datapath+'E70/SEQ5_E70_FZ_2014-05-23_11333301_z.pgm'
            cam40_b = datapath+'E40/SEQ5_E40_FZ_2014-05-23_11333601_b.pgm'
            cam40_z = datapath+'E40/SEQ5_E40_FZ_2014-05-23_11333601_z.pgm'            
            cam70_offset = -100
            cam70_factor = 0.963
            cam40_offset = -180
            cam40_factor = 0.963
            max_depth_e40 = 9651
            max_depth_e70 = 10221
            
            
        # Initializing the 5 video files
        vid_left = cv2.VideoCapture(cam_left)
        if(not(vid_left.isOpened())):
            print("Left file cannot be opened")
            sys.exit(0)
        
        vid_middle = cv2.VideoCapture(cam_middle)
        if(not(vid_middle.isOpened())):
            print("Middle file cannot be opened")
            sys.exit(0)
            
        vid_right = cv2.VideoCapture(cam_right)
        if(not(vid_right.isOpened())):
            print("Right file cannot be opened")
            sys.exit(0)
        
        vid_e70_b = readPGM(cam70_b)
        vid_e70_b.open()
        vid_e40_b = readPGM(cam40_b)
        vid_e40_b.open()
        
        vid_e70_z = readPGM(cam70_z)
        vid_e70_z.open()
        vid_e40_z = readPGM(cam40_z)
        vid_e40_z.open()
            
        # Adjusting index in each other video
        middle_index = left_index+middle_offset
        right_index = left_index+right_offset
        e70_index = int(np.round(cam70_factor*left_index+cam70_offset))
        e40_index = int(np.round(cam40_factor*left_index+cam40_offset))
        
        # Move to index and grab the frames
        vid_left.set(cv2.CAP_PROP_POS_FRAMES,left_index)
        vid_middle.set(cv2.CAP_PROP_POS_FRAMES,middle_index)
        vid_right.set(cv2.CAP_PROP_POS_FRAMES,right_index)
        ret,img_left = vid_left.read()
        ret,img_middle = vid_middle.read()
        ret,img_right = vid_right.read()
    
        # Grab radiance image of depth cameras
        img_e40 = vid_e40_b.read(e40_index)
        img_e70 = vid_e70_b.read(e70_index)
        # Normalize and convert to grayscale image
        img_e40 = (255*((img_e40-img_e40.min())/(img_e40.max()-img_e40.min()))).astype(np.uint8)
        img_e70 = (255*((img_e70-img_e70.min())/(img_e70.max()-img_e70.min()))).astype(np.uint8)
        
        # Grab DEPTH image of depth cameras
        depth_e40 = vid_e40_z.read(e40_index)
        depth_e70 = vid_e70_z.read(e70_index)
        
        
        # Release video files
        vid_left.release()
        vid_middle.release()
        vid_right.release()
        vid_e70_b.close()
        vid_e40_b.close()
        vid_e70_z.close()
        vid_e40_z.close()

        if(name=='left'):
            return img_left
        elif(name=='right'):
            return img_right
        elif(name=='middle'):
            return img_middle
        elif(name=='e40'):
            return img_e40
        elif(name=='e70'):
            return img_e70
        elif(name=='e40z'):
            return depth_e40
        elif(name=='e70z'):
            return depth_e70
        else:
            return img_left, img_middle, img_right, img_e40, img_e70, depth_e40, depth_e70
    
    
    

def displayAllImages(img_left, img_middle, img_right, img_e40, img_e70):    
    
    # Concatenate and adapt format, size and color
    pair_color = np.concatenate((img_left,img_middle,img_right),axis=1)
    pair_color = cv2.resize(pair_color,(int(0.2*pair_color.shape[1]),int(0.2*pair_color.shape[0])))

    pair_depth = np.concatenate((img_e40,img_e70),axis=1)
    ratio = pair_color.shape[1]/pair_depth.shape[1]
    pair_depth = cv2.resize(pair_depth,(int(ratio*pair_depth.shape[1]),int(ratio*pair_depth.shape[0])))
    pair_depth = cv2.cvtColor(pair_depth,cv2.COLOR_GRAY2BGR)

    pair = np.concatenate((pair_color,pair_depth),axis=0)

    cv2.imshow('all in one',pair)
    cv2.waitKey(0)
    
    
    

def predetectChessboard(calib_path=u'.//Calib//'):
    ''' This function "PRE"detects chessboard patterns in video sequences and save images for reuse 
        It's a specific function in order to use the dataset "20140523_MVcapture"
        Path variables are probably hardcoded below, except for "calib_path" which is the destination folder of the selected calibration images.
        This function should be replaced by a live stream detection in LIFE '''
    
    
    
    if os.path.isdir(calib_path):
        
        listfiles = glob.glob(os.path.join(calib_path,'*.png'))
        if(len(listfiles)>0):
            ans = input('Directory for calibration images already exists. Clean [c] or keep [k]?')
            if ans=='c':                
                for file in listfiles:
                    os.remove(file)
                print('Previous images have been removed')

    else:
    
        os.mkdir(os.path.normpath(calib_path))      
    
    
    calib_iter = 0
    
    # Frame index iteration (number of frames between each chessboard detection)
    step = 12
    # criteria used in the chessboard detection algorithm
    chessboard_detection_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    # scale factor used to resize depthcam images to facilitate chess detection
    depthcam_scalefactor = 5.0
    
    
    ## SEQUENCE 4
    fps = 25.0  # fps of the colour cams
    left_time_calibstart_seq4 = 35
    left_time_calibend_seq4 = 64
    
    # Definition of searching range
    left_pos_start = int(left_time_calibstart_seq4*fps)
    left_pos_end = int(left_time_calibend_seq4*fps)

    # Chessboard detection
    left_pos = left_pos_start
    while left_pos<left_pos_end:    # left cam index is used as reference
    
        img_left, img_middle, img_right, img_e40, img_e70, _, _ = readCamerasFrames_20140523_MVcapture(left_pos,4)
        
        # Resize to facilitate chessboard detection (see factor value in preamble)
        img_e40_rs = cv2.resize(img_e40,(int(depthcam_scalefactor*img_e40.shape[1]),int(depthcam_scalefactor*img_e40.shape[0])),0,0,cv2.INTER_NEAREST )
        img_e70_rs = cv2.resize(img_e70,(int(depthcam_scalefactor*img_e70.shape[1]),int(depthcam_scalefactor*img_e70.shape[0])),0,0,cv2.INTER_NEAREST )
    
        # Look for the chessboard corners in each image
        ret_l, corners_l = cv2.findChessboardCorners(img_left, (9,6), flags=chessboard_detection_flags )
        if ret_l:
            cv2.imwrite(os.path.join(calib_path,'%02d_left.png'%calib_iter),img_left)
        
        ret_m, corners_m = cv2.findChessboardCorners(img_middle, (9,6), flags=chessboard_detection_flags )
        if ret_m: 
            cv2.imwrite(os.path.join(calib_path,'%02d_middle.png'%calib_iter),img_middle)
            
        ret_r, corners_r = cv2.findChessboardCorners(img_right, (9,6), flags=chessboard_detection_flags )
        if ret_r: 
            cv2.imwrite(os.path.join(calib_path,'%02d_right.png'%calib_iter),img_right)
            
        ret_e70, corners_e70 = cv2.findChessboardCorners(img_e70_rs, (9,6), flags=chessboard_detection_flags )
        if ret_e70: 
            cv2.imwrite(os.path.join(calib_path,'%02d_e70.png'%calib_iter),img_e70)
            
        ret_e40, corners_e40 = cv2.findChessboardCorners(img_e40_rs, (9,6), flags=chessboard_detection_flags )
        if ret_e40: 
            cv2.imwrite(os.path.join(calib_path,'%02d_e40.png'%calib_iter),img_e40)        
    
        # If chessboard has been detected on at least one view
        if ret_l or ret_m or ret_r or ret_e40 or ret_e70:
            # increase iterator
            calib_iter += 1    
        
        print(left_pos)
        # Moving to the next image
        left_pos += step
    
    

    ## SEQUENCE 5
    fps = 25.0  # fps of the colour cams
    left_time_calibstart_seq5 = 34
    left_time_calibend_seq5 = 51
        
    # Definition of searching range
    left_pos_start = int(left_time_calibstart_seq5*fps)
    left_pos_end = int(left_time_calibend_seq5*fps)
    
    # Chessboard detection
    left_pos = left_pos_start
    while left_pos<left_pos_end:    # left cam index is used as reference
    
        img_left, img_middle, img_right, img_e40, img_e70 = readCamerasFrames_20140523_MVcapture(left_pos,5)
        
        
        # Resize to facilitate chessboard detection (see factor value in preamble)
        img_e40_rs = cv2.resize(img_e40,(int(depthcam_scalefactor*img_e40.shape[1]),int(depthcam_scalefactor*img_e40.shape[0])),0,0,cv2.INTER_NEAREST)
        img_e70_rs = cv2.resize(img_e70,(int(depthcam_scalefactor*img_e70.shape[1]),int(depthcam_scalefactor*img_e70.shape[0])),0,0,cv2.INTER_NEAREST )
    
        # Look for the chessboard corners in each image
        ret_l, corners_l = cv2.findChessboardCorners(img_left, (9,6), flags=chessboard_detection_flags )
        if ret_l:
            cv2.imwrite(os.path.join(calib_path,'%02d_left.png'%calib_iter),img_left)
        
        ret_m, corners_m = cv2.findChessboardCorners(img_middle, (9,6), flags=chessboard_detection_flags )
        if ret_m: 
            cv2.imwrite(os.path.join(calib_path,'%02d_middle.png'%calib_iter),img_middle)
            
        ret_r, corners_r = cv2.findChessboardCorners(img_right, (9,6), flags=chessboard_detection_flags )
        if ret_r: 
            cv2.imwrite(os.path.join(calib_path,'%02d_right.png'%calib_iter),img_right)
            
        ret_e70, corners_e70 = cv2.findChessboardCorners(img_e70_rs, (9,6), flags=chessboard_detection_flags )
        if ret_e70: 
            cv2.imwrite(os.path.join(calib_path,'%02d_e70.png'%calib_iter),img_e70)
            
        ret_e40, corners_e40 = cv2.findChessboardCorners(img_e40_rs, (9,6), flags=chessboard_detection_flags )
        if ret_e40: 
            cv2.imwrite(os.path.join(calib_path,'%02d_e40.png'%calib_iter),img_e40)         
    
        # If chessboard has been detected on at least one view
        if ret_l or ret_m or ret_r or ret_e40 or ret_e70:
            # increase iterator
            calib_iter += 1    
        
        print(left_pos)
        # Moving to the next image
        left_pos += step








def predetectChessboardAuto(cameras,calib_path=u'.//CalibAuto//'):
    ''' Capture and look for chessboard in cameras given by the list of dictionary "cameras"
        Chessboard calibration images are stored in "calib_path" for reuse. 
        
        "cameras" is a list of dictionary for each camera that should be used
        one camera dic should contain at least a key 'name' and a key 'id'
        (more could be added)
        
        For example: 
            cam1 = {'name':'left','id':0}
            cam2 = {'name':'right','id':1}
            cameras = [cam1,cam2]        
            
        id is the USD device id expected by cv2.VideoCapture        
        
        WARNING: This function is for now assuming all cameras have the same resolution!!!
        Adjustments will have to be made if not...                  '''
    
    
    
    if os.path.isdir(calib_path):
        
        listfiles = glob.glob(os.path.join(calib_path,'*.png'))
        if(len(listfiles)>0):
            ans = input('Directory for calibration images already exists. Clean [c] or keep [k]?')
            if ans=='c':                
                for file in listfiles:
                    os.remove(file)
                print('Previous images have been removed')

    else:
    
        os.mkdir(os.path.normpath(calib_path))            

    
    # THRESHOLDS FOR MOTION DETECTION
    # probably need to be adapted to the room, light, cameras, etc.
    
    # abs difference below this threshold are considered noise and discarded
    noise_threshold = 10
    # when the average over 30 frames (~1 sec??) of the mean of absolute differences 
    # (excluding noise defined above) between one frame and the previous is below 
    # this threshold, the scene is considered static and chessboard detection is run
    detection_threshold = 20

    # parameters for fast detection
    chessboard_detection_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK    
    
    
    
    # checking all cameras and storing handler in list 'vids'
    vids = []
    for cam in cameras:
        vids.append(cv2.VideoCapture(cam['id']))
        if(not(vids[-1].isOpened)):
            print('Error could not open Camera (%s,%d)'%(cam['name'],cam['id']))

        ret,img = vids[-1].read()
        if(not(ret)):
            print('Error could not read frames on Camera',cam['name'])
            vids.pop(-1)
        else:
            print('Camera (%s,%d): %dx%d'%(cam['name'],cam['id'],vids[-1].get(cv2.CAP_PROP_FRAME_WIDTH),vids[-1].get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
        

    
    # Displaying all cameras side-by-side, press ESC to quit or wait (about 15-20 sec)
    fr=0
    while(fr<500):
        
        img_list = []        
        for vid in vids:
            ret,img = vid.read()
            img_list.append(img)
            
        img_all = img_list[0]
        for i in range(1,len(vids)):
            img_all = np.concatenate((img_all,img_list[i]),1)
    
        fr += 1
        
        show(img_all,'Just checking!',stats=False)  
        key = cv2.waitKey(10)
        if key==27:   # if ESC is pressed, abort
            break
        
    
    
    # initializing variable used for detection of chessboards
    prev_img = np.double(img_list[0])   # previous to conmpare with
    diff_list = np.zeros(30)            # list of differences over the past 30 frames
   
    delay = 0                   # delay is added to visu time if chessboard is detected, so you can actually see it
    fr=0                        # frame counter
    frd = 0                     # frame when detection occurs
    while(fr<10000):
        
        img_list = []        
        for vid in vids:
            ret,img = vid.read()
            img_list.append(img)
    
        # diff is only calculted on one image, on the green channel for simplicity
        diff = np.abs(np.double(img_list[0][:,:,1])-prev_img[:,:,1]).astype('uint8')
        diff_list[fr%diff_list.size] = np.mean(diff[(diff>noise_threshold).nonzero()])
        prev_img = np.double(img_list[0])
    
        # average diff value over the past 30 frames
        avg_diffvalue = np.average(diff_list)
        
        # if the average is below threshold (operator standing still), then we look for chessboard
        # and if the last detection was not too recent
        if(avg_diffvalue<detection_threshold and fr>frd+50):
            
            rets = []
            cornerss = []
            for img in img_list:
                # looking for chessboard in each camera image
                ret, corners = cv2.findChessboardCorners(img, (9,6), flags=chessboard_detection_flags )
                
                rets.append(ret)
                cornerss.append(corners)
                        
            if all(rets):   # chessboard detected on each camera
                for i,cam in enumerate(cameras):
                    cv2.imwrite(os.path.join(calib_path,'%05d_%s.png'%(fr,cam['name']) ),img_list[i])
                
                # draw chessboards
                for img,corners,ret in zip(img_list,cornerss,rets):
                    cv2.drawChessboardCorners(img, (9,6), corners, ret)
                    
                # add delay to show chessboard
                delay = 500
                frd = fr
            
            
        img_all = img_list[0]
        for i in range(1,len(vids)):
            img_all = np.concatenate((img_all,img_list[i]),1)
            
                
        cv2.putText(img_all,'%f'%avg_diffvalue, (0,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
        show(img_all,'Chessboard detection. Stay still for capture and be sure chessboard is visible from every cameras',stats=False)
    
        key = cv2.waitKey(20+delay)
        if key==27:   # if ESC is pressed, abort
            break
        fr += 1
        
        delay = 0
        
        
    
        
    cv2.destroyAllWindows()
    
    for vid in vids:
        vid.release()




def calibrateMonoCamera(calibrationimages,reprojection_error_threshold_average=1.0,reprojection_error_threshold_individual=2.0,scaling_factor=1.0,refine_corners=True,interp_method=cv2.INTER_NEAREST,verbose=False,visualize=False):
    
    chessboard_detection_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    refine_corners_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    
    calibration_flags = cv2.CALIB_FIX_K3 # + cv2.CALIB_RATIONAL_MODEL
    calibration_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

    print('--------------------------------------------------------------------------------')
    print('Starting calibration of camera "%s"...'%(os.path.basename(calibrationimages[0]).split('_')[1].split('.')[0]))
    
    
    print('Calibration will be finished when:')
    print('- all calibration images have an individual reprojection error lower than %.3f AND the average reprojection error is lower than %.3f'%(reprojection_error_threshold_individual,reprojection_error_threshold_average))
    print('- OR the number of remaining calibration images is down to 10')
    
    
    max_error = 100
    avg_error = 100
    index_maxerror = None
    
    
    iteration = 0
    while( (avg_error >= reprojection_error_threshold_average or max_error>=reprojection_error_threshold_individual) and len(calibrationimages)>=10 ):
            
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in world coordinates
        imgpoints = [] # 2d points in image plane used 
        # World coordinates of chessboard corners: (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
            
        
        # Indices of images to be removed because chessboard was not found
        indices_to_pop = []
            
        
        if index_maxerror is not None:     
            if max_error>=reprojection_error_threshold_individual:
                print('Discarding image %s: Individual reprojection error is %.3f, above user defined threshold %.3f (Average error: %.3f).' % (os.path.basename(calibrationimages[index_maxerror]), max_error, reprojection_error_threshold_individual, avg_error))
            else:
                print('Discarding image %s: Average reprojection error is %.3f, above user defined threshold %.3f (Individual error: %.3f).' % (os.path.basename(calibrationimages[index_maxerror]), avg_error, reprojection_error_threshold_average, max_error))
            calibrationimages.pop(index_maxerror)
        else:
            if avg_error!=100:
                print('Average reprojection error for this iteration is %.3f, abose user defined threshold %.3f.' % (avg_error,reprojection_error_threshold_average))
            
        print('Iteration #%d: %d calibration images'%(iteration+1,len(calibrationimages)))
        
        for imageindex,imagefile in enumerate(calibrationimages):
            
            img = cv2.imread(imagefile)
            (w,h) = img.shape[1::-1] # original image format
        
            if(scaling_factor!=1.):    
                # Resize to facilitate chessboard detection (see factor value in preamble)
                img = cv2.resize(img,(int(scaling_factor*img.shape[1]),int(scaling_factor*img.shape[0])),0,0,interp_method )        
            
            ret, corners = cv2.findChessboardCorners(img, (9,6), flags=chessboard_detection_flags )
            
            if ret: # if found
                if verbose:
                    print('Chessboard found in %s'%os.path.basename(imagefile))
                
                if(refine_corners==True):
                    # refine the corners positions
                    cv2.cornerSubPix(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),corners,(int(img.shape[0]/100),int(img.shape[0]/100)),(-1,-1),refine_corners_criteria)
                
                if visualize:
                    # draw the corners on the image
                    cv2.drawChessboardCorners(img, (9,6), corners, ret)            
                    # add text for debug
                    textposition = (0,int(img.shape[0]/16))
                    textfont = cv2.FONT_HERSHEY_PLAIN
                    textsize = int(img.shape[0]/180)
                    textcolor = (0,0,255)
                    textthickness = int(img.shape[0]/120)
                    cv2.putText(img,'-'.join(os.path.basename(imagefile).split('_')[:2]), textposition, textfont, textsize, textcolor, textthickness)
                    
                    # show result of the detection
                    win = show(img,'Chessboard detection',width=800, stats=False)
                    cv2.waitKey(800)    # display each image for 0.8 sec
                    
                if(scaling_factor!=1.):
                    # Rescaling corners positions too
                    for k in range(len(corners)):
                        corners[k] /= scaling_factor
                
                # save the corners positions
                imgpoints.append(corners)
                objpoints.append(objp)  # append reference worls coordinates as many times
                
            else:
                if verbose:
                    print('Chessboard NOT found in %s.'%os.path.basename(imagefile))
                indices_to_pop.append(imageindex)
                
        for i in indices_to_pop[::-1]:
            calibrationimages.pop(i)
        
        ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h), None, None, flags=calibration_flags, criteria=calibration_criteria)
        
        index_maxerror, max_error, avg_error = reprojectToCalibrationImage(cameraMatrix, distCoeffs, rvecs, tvecs, objpoints, imgpoints)
        if verbose:
            print('Maximum reprojection error %f found for image %s (Reprojection error: %f)'%(max_error, os.path.basename(calibrationimages[index_maxerror]), avg_error))
        
        iteration += 1
        
    if(avg_error < reprojection_error_threshold_average and max_error < reprojection_error_threshold_individual):
        print('Success! Camera "%s" has been calibrated using %d images.'%(os.path.basename(calibrationimages[0]).split('_')[1].split('.')[0],len(calibrationimages)))
        print('Average reprojection error: %.3f. Individual max reprojection error: %.3f'%(ret,max_error))
    else:
        print('WARNING: Max number of iteration has been reached. Check the results!')
        print('Camera "%s" has been calibrated using %d images.'%(os.path.basename(calibrationimages[0]).split('_')[1].split('.')[0],len(calibrationimages)))
        if(ret):
            print('Average reprojection error: %.3f. Individual max reprojection error: %.3f'%(ret,max_error))
          
    if visualize:
        cv2.destroyWindow(win)
            
            
    return cameraMatrix, distCoeffs, imgpoints, calibrationimages



def calibrateStereoCamera( cam1, cam2, calib_path, scaling_factor1=None, scaling_factor2=None, refine_corners1=True, refine_corners2=True):
    
    print('--------------------------------------------------------------------------------')
    
    calibration_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.00001)
    
    if cam1[-4:]=='.npz':
        name1 = cam1
    else:
        name1 = 'calib_%s.npz'%cam1
        
    calib_file1 = os.path.join(calib_path,name1)
    if os.path.isfile(calib_file1) :
        # Loading intrinsec parameters from individual parameters if file exists
        calib1 = np.load(calib_file1)    
        cameraMatrix1 = calib1['K']
        distCoeffs1 = calib1['k']
        imgpoints1 = calib1['imgpoints']
        calibrationimages1 = calib1['calibrationimages']
    else:        
        # Performing individual calibration if not
        calibrationimages = getCalibrationImages(calib_path,cam1)
        cameraMatrix1, distCoeffs1, imgpoints1, calibrationimages1 = calibrateMonoCamera(calibrationimages,scaling_factor1,refine_corners1)

    if cam2[-4:]=='.npz':
        name2 = cam2
    else:
        name2 = 'calib_%s.npz'%cam2
    
    calib_file2 = os.path.join(calib_path,name2)
    if os.path.isfile(calib_file2) :
        # Loading intrinsec parameters from individual parameters if file exists    
        calib2 = np.load(calib_file2)
        cameraMatrix2 = calib2['K']
        distCoeffs2 = calib2['k']
        imgpoints2 = calib2['imgpoints']
        calibrationimages2 = calib2['calibrationimages']
    else:
        # Performing individual calibration if not        
        calibrationimages = getCalibrationImages(calib_path,cam2)
        cameraMatrix2, distCoeffs2, imgpoints2, calibrationimages2 = calibrateMonoCamera(calibrationimages,scaling_factor2,refine_corners2)    
    
    # Selecting point coordinates corresponding to calibration images that are common to both cameras individual calibration
    indices1 = []
    for f in calibrationimages1:
        indices1.append(int(os.path.basename(f)[:2])) 
    
    indices2 = []
    for f in calibrationimages2:
        indices2.append(int(os.path.basename(f)[:2])) 

    sub = [val for val in indices1 if val in indices2]

    imgpoints1 = [imgpoints1[j] for j in [indices1.index(i) for i in sub]]
    imgpoints2 = [imgpoints2[j] for j in [indices2.index(i) for i in sub]]
    

    # debug: check that the right images are selected
#    for k in range(len(sub)):
#    
#        img1 = cv2.imread(u'.//Calib//%02d_%s.png'%(sub[k],cam1))
#        imgpt1 = imgpoints1[k]
#        img1 = cv2.drawChessboardCorners(img1, (9,6), imgpt1, 1)
#        
#        img2 = cv2.imread(u'.//Calib//%02d_%s.png'%(sub[k],cam2))
#        img2 = cv2.resize(img2,(int(4.5*img2.shape[1]),int(4.5*img2.shape[0])),0,0,cv2.INTER_NEAREST )
#        imgpt2 = imgpoints2[k]
#        imgpt2 = 4.5*imgpt2
#        img2 = cv2.drawChessboardCorners(img2, (9,6), imgpt2, 1)
#        
#        img = np.concatenate((cv2.resize(img1,(int(0.5*img1.shape[1]),int(0.5*img1.shape[0])),0,0,cv2.INTER_NEAREST ),img2),1)
#        show(img,'test'+str(k))
#        
#    cv2.waitKey(0)
#    
#    cv2.destroyWindow('test1')
#    cv2.destroyWindow('test2')
    

    
    # World coordinates of chessboard corners: (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objpoints = [objp for k in range(len(imgpoints1))]
    
    
    ret, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (0,0), flags=cv2.CALIB_FIX_INTRINSIC, criteria=calibration_criteria)
    print('Stereo calibration of cameras "%s" and "%s" done with %d calibration images. Reprojection error: %f'%(os.path.basename(calibrationimages1[0]).split('_')[1],os.path.basename(calibrationimages2[0]).split('_')[1],len(imgpoints1),ret))
    

    

    return cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F



def getCalibrationImages(path,camera_name):
    ''' Return the list of calibration imagefiles concerning camera1, or both camera1 and camera2 if camera2 is provided
        camera1 and camera2 are string identifying the camera in the calibration image filenames '''
    
    
    rawlist = glob.glob(os.path.join(path,'*'+camera_name+'*.png'))
    abslist = []
    for f in rawlist:
        abslist.append(os.path.abspath(f))
        
    return abslist


