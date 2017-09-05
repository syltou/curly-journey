import pgmTools
import os
import struct
import numpy as np
import cv2
import glob
from pgmTools import pgmVideo


# files and params

path = '//personal.mh.se/data/konton/svl/s/sylvto/Work/Data/20140523_MVcapture/'
path = 'C:/20140523_MVcapture/'

out_path = os.path.join('.','Samples')


##### SEQUENCE #4
### color cams
##cam_left = path+'Left/SEQ4_LEFT_MVI_0393.MOV'
##cam_middle = path+'Middle/SEQ4_MIDDLE_MVI_0003.MOV'
##cam_right = path+'Right/SEQ4_RIGHT_MVI_0567.MOV'
##middle_offset = int(+22)
##right_offset = int(-30)
##left_time_calibstart = 35
##left_time_calibend = 64
### depth cams
##cam70_b = path+'E70/SEQ4_E70_FZ_2014-05-23_11250601_b.pgm'
##cam70_z = path+'E70/SEQ4_E70_FZ_2014-05-23_11250601_z.pgm'
##cam40_b = path+'E40/SEQ4_E40_FZ_2014-05-23_11250701_b.pgm'
##cam40_z = path+'E40/SEQ4_E40_FZ_2014-05-23_11250701_z.pgm'
##cam70_offset = -178
##cam70_factor = 0.963
##cam40_offset = -212
##cam40_factor = 0.963
##suff = 'SEQ4'
##max_depth_e40 = 9643
##max_depth_e70 = 10207

### SEQUENCE #5
# color cams
cam_left = path+'Left/SEQ5_LEFT_MVI_0394.MOV'
cam_middle = path+'Middle/SEQ5_MIDDLE_MVI_0004.MOV'
cam_right = path+'Right/SEQ5_RIGHT_MVI_0568.MOV'
middle_offset = int(+27)
right_offset = int(+62)
left_time_calibstart = 34
left_time_calibend = 51
# depth cams
cam70_b = path+'E70/SEQ5_E70_FZ_2014-05-23_11333301_b.pgm'
cam70_z = path+'E70/SEQ5_E70_FZ_2014-05-23_11333301_z.pgm'
cam40_b = path+'E40/SEQ5_E40_FZ_2014-05-23_11333601_b.pgm'
cam40_z = path+'E40/SEQ5_E40_FZ_2014-05-23_11333601_z.pgm'
cam70_offset = -100
cam70_factor = 0.963
cam40_offset = -180
cam40_factor = 0.963
suff = 'SEQ5'
max_depth_e40 = 9651
max_depth_e70 = 10221


## VideoCapture objects
vid_left = cv2.VideoCapture(cam_left)
vid_middle = cv2.VideoCapture(cam_middle)
vid_right = cv2.VideoCapture(cam_right)

dep_e70 = pgmVideo(cam70_z)
dep_e70.readHeader()
dep_e40 = pgmVideo(cam40_z)
dep_e40.readHeader()
rad_e70 = pgmVideo(cam70_b)
rad_e70.readHeader()
rad_e40 = pgmVideo(cam40_b)
rad_e40.readHeader()


# Sample frames selection (left video index)
frames = [1300]#range(1300,3500,200)



for left_index in frames:

    print(left_index)

    # Synchronization of sequences indices
    middle_index = left_index+middle_offset
    right_index = left_index+right_offset
    e70_index = int(np.round(cam70_factor*left_index+cam70_offset))
    e40_index = int(np.round(cam40_factor*left_index+cam40_offset))

    # Move pointers and grab the video frames
    vid_left.set(cv2.CAP_PROP_POS_FRAMES,left_index)
    ret,img_l = vid_left.read()
    vid_middle.set(cv2.CAP_PROP_POS_FRAMES,middle_index)
    ret,img_m = vid_middle.read()
    vid_right.set(cv2.CAP_PROP_POS_FRAMES,right_index)
    ret,img_r = vid_right.read()

    # Grab depth images, change range and flip
    imgz_e70_raw = dep_e70.getFrame(e70_index)
    imgz_e70 = np.fliplr((255*(imgz_e70_raw/max_depth_e70)).astype(np.uint8))
    imgz_e70 = cv2.cvtColor(imgz_e70,cv2.COLOR_GRAY2RGB)
    #imgz_e70 = cv2.resize(imgz_e70,(int(5.0*imgz_e70.shape[1]),int(5.0*imgz_e70.shape[0])),0,0,cv2.INTER_NEAREST )
    imgz_e40_raw = dep_e40.getFrame(e40_index)
    imgz_e40 = np.fliplr((255*(imgz_e40_raw/max_depth_e40)).astype(np.uint8))
    imgz_e40 = cv2.cvtColor(imgz_e40,cv2.COLOR_GRAY2RGB)
    #imgz_e40 = cv2.resize(imgz_e40,(int(5.0*imgz_e40.shape[1]),int(5.0*imgz_e40.shape[0])),0,0,cv2.INTER_NEAREST )

    # Grab radiance images, change range and flip
    imgb_e70_raw = rad_e70.getFrame(e70_index)
    imgb_e70 = np.fliplr((255*(imgb_e70_raw/1024)).astype(np.uint8))
    imgb_e70 = cv2.cvtColor(imgb_e70,cv2.COLOR_GRAY2RGB)
    #imgb_e70 = cv2.resize(imgb_e70,(int(5.0*imgb_e70.shape[1]),int(5.0*imgb_e70.shape[0])),0,0,cv2.INTER_NEAREST )
    imgb_e40_raw = rad_e40.getFrame(e40_index)
    imgb_e40 = np.fliplr((255*(imgb_e40_raw/1024)).astype(np.uint8))
    imgb_e40 = cv2.cvtColor(imgb_e40,cv2.COLOR_GRAY2RGB)
    #imgb_e40 = cv2.resize(imgb_e40,(int(5.0*imgz_e40.shape[1]),int(5.0*imgb_e40.shape[0])),0,0,cv2.INTER_NEAREST )
    

    
    
    # Combine all in one
    three_color = np.concatenate((img_l,img_m,img_r),axis=0)    # 1920x3240
    three_color = cv2.resize(three_color,(int(0.2*three_color.shape[1]),int(0.2*three_color.shape[0])))    # 384*648

    pair_depth = np.concatenate((imgz_e70,imgz_e40),axis=0)       # 160x240
    
    ratio = three_color.shape[0]/pair_depth.shape[0]
    pair_depth = cv2.resize(pair_depth,(int(ratio*pair_depth.shape[1]),int(ratio*pair_depth.shape[0]))) # 432x648

    allinone = np.concatenate((three_color,pair_depth),axis=1)  # 816x648

    # Mark frames
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(allinone,'LEFT',(10,206), font, 1,(255,255,255),3,cv2.LINE_AA)
    cv2.putText(allinone,'MID',(10,422), font, 1,(255,255,255),3,cv2.LINE_AA)
    cv2.putText(allinone,'RIGHT',(10,638), font, 1,(255,255,255),3,cv2.LINE_AA)
    cv2.putText(allinone,'E70',(394,314), font, 1,(255,255,255),3,cv2.LINE_AA)
    cv2.putText(allinone,'E40',(394,638), font, 1,(255,255,255),3,cv2.LINE_AA)
    

    # Save images
    outfile_left = os.path.join(out_path,suff+'_LEFT_fr%d.bmp'%(left_index))
    outfile_middle = os.path.join(out_path,suff+'_MIDDLE_fr%d.bmp'%(left_index))
    outfile_right = os.path.join(out_path,suff+'_RIGHT_fr%d.bmp'%(left_index))
    outfile_e40_z = os.path.join(out_path,suff+'_E40_z_fr%d.bmp'%(left_index))
    outfile_e70_z = os.path.join(out_path,suff+'_E70_z_fr%d.bmp'%(left_index))
    outfile_mix = os.path.join(out_path,suff+'_MIX_fr%d.bmp'%(left_index))
    outfile_e40_b = os.path.join(out_path,suff+'_E40_b_fr%d.bmp'%(left_index))
    outfile_e70_b = os.path.join(out_path,suff+'_E70_b_fr%d.bmp'%(left_index))

    outfile_e40_z_pgm = os.path.join(out_path,suff+'_E40_z_fr%d.pgm'%(left_index))
    pgmWrite_e40_z = pgmVideo(outfile_e40_z_pgm)
    pgmWrite_e40_z.height = 120
    pgmWrite_e40_z.width  = 160
    outfile_e70_z_pgm = os.path.join(out_path,suff+'_E70_z_fr%d.pgm'%(left_index))
    pgmWrite_e70_z = pgmVideo(outfile_e70_z_pgm)
    pgmWrite_e70_z.height = 120
    pgmWrite_e70_z.width  = 160
    outfile_e40_b_pgm = os.path.join(out_path,suff+'_E40_b_fr%d.pgm'%(left_index))
    pgmWrite_e40_b = pgmVideo(outfile_e40_b_pgm)
    pgmWrite_e40_b.height = 120
    pgmWrite_e40_b.width  = 160
    outfile_e70_b_pgm = os.path.join(out_path,suff+'_E70_b_fr%d.pgm'%(left_index))
    pgmWrite_e70_b = pgmVideo(outfile_e70_b_pgm)
    pgmWrite_e70_b.height = 120
    pgmWrite_e70_b.width  = 160
    

    cv2.imwrite(outfile_left,img_l)
    cv2.imwrite(outfile_middle,img_m)
    cv2.imwrite(outfile_right,img_r)
    cv2.imwrite(outfile_e40_z,imgz_e40)
    cv2.imwrite(outfile_e70_z,imgz_e70)
    cv2.imwrite(outfile_mix,allinone)

    cv2.imwrite(outfile_e40_b,imgb_e40)
    cv2.imwrite(outfile_e70_b,imgb_e70)


    #cv2.imwrite(outfile_e40_z_pgm,imgz_e40_raw)
    #cv2.imwrite(outfile_e70_z_pgm,imgz_e70_raw)
    #cv2.imwrite(outfile_e40_b_pgm,imgb_e40_raw)
    #cv2.imwrite(outfile_e70_b_pgm,imgb_e70_raw)
    pgmWrite_e40_z.writeFrame(0,imgz_e40_raw)
    pgmWrite_e70_z.writeFrame(0,imgz_e70_raw)
    pgmWrite_e40_b.writeFrame(0,imgb_e40_raw)
    pgmWrite_e70_b.writeFrame(0,imgb_e70_raw)
    
    




vid_left.release()
vid_middle.release()
vid_right.release()

