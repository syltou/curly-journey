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
path = '//ID20809271/LFdata/'



### SEQUENCE #4
# color cams
cam_left = path+'Left/SEQ4_LEFT_MVI_0393.MOV'
cam_middle = path+'Middle/SEQ4_MIDDLE_MVI_0003.MOV'
cam_right = path+'Right/SEQ4_RIGHT_MVI_0567.MOV'
middle_offset = int(+22)
right_offset = int(-30)
left_time_calibstart = 35
left_time_calibend = 64
# depth cams
cam70_b = path+'E70/SEQ4_E70_FZ_2014-05-23_11250601_b.pgm'
cam70_z = path+'E70/SEQ4_E70_FZ_2014-05-23_11250601_z.pgm'
cam40_b = path+'E40/SEQ4_E40_FZ_2014-05-23_11250701_b.pgm'
cam40_z = path+'E40/SEQ4_E40_FZ_2014-05-23_11250701_z.pgm'
cam70_offset = -178
cam70_factor = 0.963
cam40_offset = -212
cam40_factor = 0.963
suff = 'SEQ4'
max_depth_e40 = 9643
max_depth_e70 = 10207

##### SEQUENCE #5
### color cams
##cam_left = path+'Left/SEQ5_LEFT_MVI_0394.MOV'
##cam_middle = path+'Middle/SEQ5_MIDDLE_MVI_0004.MOV'
##cam_right = path+'Right/SEQ5_RIGHT_MVI_0568.MOV'
##middle_offset = int(+27)
##right_offset = int(+62)
##left_time_calibstart = 34
##left_time_calibend = 51
### depth cams
##cam70_b = path+'E70/SEQ5_E70_FZ_2014-05-23_11333301_b.pgm'
##cam70_z = path+'E70/SEQ5_E70_FZ_2014-05-23_11333301_z.pgm'
##cam40_b = path+'E40/SEQ5_E40_FZ_2014-05-23_11333601_b.pgm'
##cam40_z = path+'E40/SEQ5_E40_FZ_2014-05-23_11333601_z.pgm'
##cam70_offset = -100
##cam70_factor = 0.963
##cam40_offset = -180
##cam40_factor = 0.963
##suff = 'SEQ5'
##max_depth_e40 = 9651
##max_depth_e70 = 10221


## VideoCapture objects
vid_left = cv2.VideoCapture(cam_left)
vid_middle = cv2.VideoCapture(cam_middle)
vid_right = cv2.VideoCapture(cam_right)

vid_e70 = pgmVideo(cam70_z)
vid_e70.readHeader()
vid_e40 = pgmVideo(cam40_z)
vid_e40.readHeader()


# Sample frames selection (left video)
frame_start = 700
frame_end = 850


fourcc = cv2.VideoWriter_fourcc(*'MSVC')
## VideoWriter objects
out_left = cv2.VideoWriter()
outfile_left = suff+'_LEFT_FR%d-%d.avi'%(frame_start,frame_end)
out_left.open(outfile_left,fourcc, 25.0, (int(vid_left.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid_left.get(cv2.CAP_PROP_FRAME_HEIGHT))))

out_middle = cv2.VideoWriter()
outfile_middle = suff+'_MIDDLE_FR%d-%d.avi'%(frame_start,frame_end)
out_middle.open(outfile_middle,fourcc, 25.0, (int(vid_middle.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid_middle.get(cv2.CAP_PROP_FRAME_HEIGHT))))

out_right = cv2.VideoWriter()
outfile_right = suff+'_RIGHT_FR%d-%d.avi'%(frame_start,frame_end)
out_right.open(outfile_right,fourcc, 25.0, (int(vid_right.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid_right.get(cv2.CAP_PROP_FRAME_HEIGHT))))

out_e40 = cv2.VideoWriter()
outfile_e40 = suff+'_E40_FR%d-%d.avi'%(frame_start,frame_end)
out_e40.open(outfile_e40,fourcc, 25.0, (vid_e40.width,vid_e40.height))

out_e70 = cv2.VideoWriter()
outfile_e70 = suff+'_E70_FR%d-%d.avi'%(frame_start,frame_end)
out_e70.open(outfile_e70,fourcc, 25.0, (vid_e70.width,vid_e70.height))

out_mix = cv2.VideoWriter()
outfile_mix = suff+'_MIX_FR%d-%d.avi'%(frame_start,frame_end)
out_mix.open(outfile_mix,fourcc, 25.0, ( int(0.2*vid_right.get(cv2.CAP_PROP_FRAME_WIDTH)) + int(0.3*vid_right.get(cv2.CAP_PROP_FRAME_HEIGHT)*vid_e40.width/vid_e40.height),
                                       int(0.6*vid_right.get(cv2.CAP_PROP_FRAME_HEIGHT)) ))



for left_index in range(frame_start,frame_end):

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
    img_e70 = vid_e70.getFrame(e70_index)
    img_e70 = np.fliplr((255*(img_e70/max_depth)).astype(np.uint8))
    img_e70 = cv2.cvtColor(img_e70,cv2.COLOR_GRAY2RGB)
    #img_e70 = cv2.resize(img_e70,(int(5.0*img_e70.shape[1]),int(5.0*img_e70.shape[0])),0,0,cv2.INTER_NEAREST )
    img_e40 = vid_e40.getFrame(e40_index)
    img_e40 = np.fliplr((255*(img_e40/max_depth)).astype(np.uint8))
    img_e40 = cv2.cvtColor(img_e40,cv2.COLOR_GRAY2RGB)
    #img_e40 = cv2.resize(img_e40,(int(5.0*img_e40.shape[1]),int(5.0*img_e40.shape[0])),0,0,cv2.INTER_NEAREST )
    


    # Write frames in their respective files
    out_left.write(img_l)
    out_middle.write(img_m)
    out_right.write(img_r)
    out_e70.write(img_e70)
    out_e40.write(img_e40)
    
    # Combine all in one
    three_color = np.concatenate((img_l,img_m,img_r),axis=0)    # 1920x3240
    three_color = cv2.resize(three_color,(int(0.2*three_color.shape[1]),int(0.2*three_color.shape[0])))    # 384*648

    pair_depth = np.concatenate((img_e70,img_e40),axis=0)       # 160x240
    
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
    
    out_mix.write(allinone)



out_left.release()
out_middle.release()
out_right.release()
out_e70.release()
out_e40.release()
out_mix.release()


vid_left.release()
vid_middle.release()
vid_right.release()

