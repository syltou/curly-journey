# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:51:15 2017

@author: sylvto
"""



import cv2
from miscTools import id_generator
import os, re, struct
import numpy as np



def show( img, winName=None, norm=True, stats=True, scale=1.0, height=None, width=None, position=(0,0), interp=cv2.INTER_LINEAR, shownan=False):

    if winName is None:        
        winName = '['+id_generator()+']' 
    winName = winName + ' %dx%d pixels'%(img.shape[1], img.shape[0])
        
    if scale!=1 :
        img = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])),0,0,interp )
    else:
        if height is not None:
            scale = height/img.shape[0]
            img = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])),0,0,interp )
        else:
            if width is not None:
                scale = width/img.shape[1]
                img = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])),0,0,interp )


    if stats:
        winName = winName + '%s, %d channels, range: [%.1f,%.1f] %s' % (
                        ' (Zoom: %dx%d)'%(img.shape[1],img.shape[0]) if (scale!=1 or height!=None or width!=None) else '',
                        img.shape[2] if len(img.shape)>2 else 1,
                        np.nanmin(img), np.nanmax(img),
                        'normalized to %s for display.'%('[0,255]' if img.dtype=='uint8' else '[0,1]') if norm==True
                            else 'values outside %s are not displayed.'%('[0,255]' if img.dtype=='uint8' else '[0,1]') )    
    

    if shownan:
        if(len(img.shape)==2):
            img=np.concatenate((img[:,:,np.newaxis],img[:,:,np.newaxis],img[:,:,np.newaxis]),2)
        
        nanindex = np.argwhere(np.isnan(img))
        if(len(nanindex)>0):
            img[nanindex[:,0],nanindex[:,1],2]=np.nanmax(img)
            
        winName = winName + ' NaN values are shown in red.'

            
    if norm:
        img = normalize(img)
    
        
    ret = cv2.getWindowProperty(winName,cv2.WND_PROP_AUTOSIZE)
    if(ret==-1):
        cv2.namedWindow(winName,flags=cv2.WINDOW_AUTOSIZE)
        print('Figure:',winName)
    
    cv2.moveWindow(winName,position[0],position[1])
    cv2.imshow(winName,img)
    
    return winName


def displayAll():
    
    cv2.waitKey(0)
    


    
def normalize(img,new_min=0,new_max=1):

    img = (img - np.nanmin(img))/(np.nanmax(img)-np.nanmin(img))
    img = img*(new_max-new_min)+new_min
    return img
    



class readPGM:

    fullpath = ''
    fileh = None
    
    width = 0
    height = 0
    maxval = 0

    index = 0
    nbframes = 1

    # values in bytes here
    filesize = 0
    offset = 0
    elementsize = 0
    imagesize = 0


    def __init__(self,fullpath,verbose=False):

        self.fullpath = fullpath

        if(os.path.isfile(fullpath)):
            pass
        else:
            print("File %s does not exist!" % os.path.abspath(fullpath))
            return None
        
        
        # Read header to get variables
        with open(fullpath, 'rb') as f:
            buffer = f.read(100)
            
        try:
            header, width, height, maxval = re.search(
                b"(^P5\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
        except AttributeError:
            raise ValueError("File %s is not a raw PGM file!" % os.path.abspath(fullpath))
            
            
        self.width  = int(width)
        self.height = int(height)
        self.maxval = int(maxval)
        
        self.offset = len(header)
        self.filesize = os.stat(fullpath).st_size

        self.elementsize = 1 if self.maxval<256 else 2
        self.imagesize = self.elementsize * self.width * self.height
        self.nbframes = int(self.filesize / (self.imagesize + self.offset))

        if(verbose):
            print('PGM file %s'%self.fullpath)
            print('width=%d, height=%d, nbframes=%d, max=%d'%(self.width,self.height,self.nbframes,self.maxval))


    def open(self):
        if(self.fileh is None):
            self.fileh = open(self.fullpath, 'rb')
        else:
            if(self.fileh.closed):
                self.fileh = open(self.fullpath, 'rb')
            else:
                print('[imgTools] in readPGM::open(), file is already opened')

    def close(self):
        if(self.fileh.closed):
            print('[imgTools] in readPGM::close(), file is already closed')
        else:
            self.fileh.close()


    def getNbFrames(self):
        return self.nbframes

    def getImgSize(self):
        return (self.width,self.height)

    def getHeight(self):
        return self.height

    def getWidth(self):
        return self.width

    def getMaxval(self):
        return self.maxval

    def getIndex(self):
        return self.index

    def setIndex(self,newindex):

        # check index value
        if(newindex<0):
            print('[imgTools] in readPGM::setIndex(), index value (%d) cannot be negative, setting index=0 instead.' % newindex)
            newindex=0
            
        if(newindex>=self.nbframes):
            print('[imgTools] in readPGM::setIndex(), index value (%d) is larger than the number of images in the file (%d), looping through file instead (new index value = %d).' % (newindex, self.nbframes, newindex%self.nbframes))
            newindex=newindex%self.nbframes

        if(newindex<self.index):
            self.index = 0
            self.fileh.close()
            self.fileh = open(self.fullpath, 'rb')
            
        ### super low and stupid
#        while(self.index<newindex):
#            self.readNext()

        ### try with this
        self.fileh.seek( (self.offset+self.imagesize)*(newindex-self.index) )
        self.index = newindex


    def readNext(self):  
        ''' reads the next frame '''
            
        # check index value
        if(self.index>=self.nbframes):
            print('[imgTools] in readPGM::readNext(), index value (%d) reached the end of the file (%d frames), looping back to first image.' % (self.index, self.nbframes))
            self.index = 0
            self.fileh.close()
            self.fileh = open(self.fullpath, 'rb')
        

        # pass header
        self.fileh.read(self.offset)
        # read frame data
        imagedata = self.fileh.read(self.imagesize)
        # unpack data
        image = struct.unpack('>'+'h'*self.width*self.height, imagedata)
        image = np.array(image,dtype=np.uint8 if self.elementsize==1 else np.uint16).reshape((self.height,self.width))
        image = np.fliplr(image)
        
        self.index += 1

        return image 
    
    
    def read(self,index):
        ''' generic read function '''
        
        self.setIndex(index)
        return self.readNext()





def warpDepthMap(DepthMap, K_input, Rt_input, K_output, Rt_output, WIDTH_output, HEIGHT_output, maxDepthValue=None):
    
    if(maxDepthValue is None):
        maxDepthValue = DepthMap.max()
    
    # projection matrices
    P_input = np.dot(K_input,Rt_input)
    P_output = np.dot(K_output,Rt_output)
    
    R_input = Rt_input[:,0:3]
    T_input = Rt_input[:,3]
    R_output = Rt_output[:,0:3]
    T_output = Rt_output[:,3]
    
    (HEIGHT_input,WIDTH_input) = DepthMap.shape
    
    warpedDepthMap = np.zeros((HEIGHT_output,WIDTH_output),dtype=np.uint8)
    
    
    for x_input in np.arange(WIDTH_input):
        for y_input in np.arange(HEIGHT_input):            
            
            # coord in original cam 
            point_input = np.matrix([[x_input],[y_input],[1.]])

            # calculate world coord
            wcoords = np.matrix([[0],[0],[0],[0]])
            zetaVector = np.dot( R_input.I, np.dot( K_input.I, point_input) )
            lambdaScalar = ((DepthMap[y_input,x_input] - -T_input[2])/zetaVector[2])[0,0]
            wcoords[0:3,0] = -T_input + np.dot( lambdaScalar*R_input.I, np.dot( K_input.I, point_input) )
            wcoords[3,0] = 1.      # homogeneous
            # target coord
            point_output = np.dot(P_output,wcoords)
            point_output = point_output/point_output[2]
            
            x_output = np.round(point_output[0,0])
            y_output = np.round(point_output[1,0])
            
            
            #print('%d,%d => %d,%d'%(x_input,y_input,x_output,y_output))
            
            if(x_output<WIDTH_output and x_output>=0 and y_output<HEIGHT_output and y_output>=0):
                warpedDepthMap[int(y_output),int(x_output)] = 255.*DepthMap[y_input,x_input]/maxDepthValue
                
    return warpedDepthMap
            
            




#def detectChessboard_old():
#    ''' This function detects chessboard patterns in video sequences and save images for reuse
#        It's a specific function in order to use the dataset "20140523_MVcapture"
#        All variables are hardcoded below.
#        This function should be replaced by a live stream detection in LIFE '''
#    
#    
#    ## files and params
#    #datapath = '//personal.mh.se/data/konton/svl/s/sylvto/Work/Data/20140523_MVcapture/'
#    #datapath = 'C:/20140523_MVcapture/'
#    datapath = u'//ID20809271/LFdata/'
#    calib_path = u'.//Calib//'    
#
#    calib_iter = 0
#    
#    # Frame index iteration (number of frames between each chessboard detection)
#    step = 12
#    # criteria used in the chessboard detection algorithm
#    chess_detection_flags = cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_ADAPTIVE_THRESH 
#    # scale factor used to resize depthcam images to facilitate chess detection
#    depthcam_scalefactor = 5.0
#    
#    
#    #__________________________________________________________________________
#    # SEQUENCE #4
#    # color cams
#    cam_left = datapath+'Left/SEQ4_LEFT_MVI_0393.MOV'
#    cam_middle = datapath+'Middle/SEQ4_MIDDLE_MVI_0003.MOV'
#    cam_right = datapath+'Right/SEQ4_RIGHT_MVI_0567.MOV'
#    middle_offset = int(+22)
#    right_offset = int(-30)
#    left_time_calibstart = 35
#    left_time_calibend = 64
#    # depth cams (but radiance sequences)
#    cam70_b = datapath+'E70/SEQ4_E70_FZ_2014-05-23_11250601_b.pgm'
#    cam40_b = datapath+'E40/SEQ4_E40_FZ_2014-05-23_11250701_b.pgm'
#    cam70_offset = -178
#    cam70_factor = 0.963
#    cam40_offset = -212
#    cam40_factor = 0.963
#    
#    
#    # Initializing the 5 video files
#    vid_left = cv2.VideoCapture(cam_left)
#    if(not(vid_left.isOpened())):
#        print("Left file cannot be opened")
#        exit(0)
#    
#    vid_middle = cv2.VideoCapture(cam_middle)
#    if(not(vid_middle.isOpened())):
#        print("Middle file cannot be opened")
#        exit(0)
#        
#    vid_right = cv2.VideoCapture(cam_right)
#    if(not(vid_right.isOpened())):
#        print("Right file cannot be opened")
#        exit(0)
#    
#    vid_e70 = readPGM(cam70_b)
#    vid_e70.open()
#    vid_e40 = readPGM(cam40_b)
#    vid_e40.open()
#    
#    # Displaying media information
#    nbf_left, fps_left, w_left, h_left = vid_left.get(cv2.CAP_PROP_FRAME_COUNT), vid_left.get(cv2.CAP_PROP_FPS), vid_left.get(cv2.CAP_PROP_FRAME_WIDTH), vid_left.get(cv2.CAP_PROP_FRAME_HEIGHT)
#    print('Left video:',w_left,'x',h_left,'-',nbf_left,'frames @', fps_left,'fps.')
#    nbf_middle, fps_middle, w_middle, h_middle = vid_middle.get(cv2.CAP_PROP_FRAME_COUNT), vid_middle.get(cv2.CAP_PROP_FPS), vid_middle.get(cv2.CAP_PROP_FRAME_WIDTH), vid_middle.get(cv2.CAP_PROP_FRAME_HEIGHT)
#    print('Middle video:',w_middle,'x',h_middle,'-',nbf_middle,'frames @', fps_middle,'fps.')
#    nbf_right, fps_right, w_right, h_right = vid_right.get(cv2.CAP_PROP_FRAME_COUNT), vid_right.get(cv2.CAP_PROP_FPS), vid_right.get(cv2.CAP_PROP_FRAME_WIDTH), vid_right.get(cv2.CAP_PROP_FRAME_HEIGHT)
#    print('Right video:',w_right,'x',h_right,'-',nbf_right,'frames @', fps_right,'fps.')
#    
#    print('Depth video E70:',vid_e70.width,'x',vid_e70.height,'-',vid_e70.nbframes,'frames @', fps_left*cam70_factor,'fps.')
#    print('Depth video E40:',vid_e40.width,'x',vid_e40.height,'-',vid_e40.nbframes,'frames @', fps_left*cam40_factor,'fps.')
#    
#    
#    # Chessboard detection
#    # Definition of searching range
#    left_pos_start = int(left_time_calibstart*fps_left)
#    left_pos_end = int(left_time_calibend*fps_left)
#
#    
#    left_pos = left_pos_start
#    while left_pos<left_pos_end:    # left cam index is used as reference
#    
#        # Adjusting index in each other video
#        middle_pos = left_pos+middle_offset
#        right_pos = left_pos+right_offset
#        e70_pos = int(np.round(cam70_factor*left_pos+cam70_offset))
#        e40_pos = int(np.round(cam40_factor*left_pos+cam40_offset))
#        print(left_pos,middle_pos,right_pos,e40_pos,e70_pos)
#    
#    
#        # Move to index and grab the frames
#        vid_left.set(cv2.CAP_PROP_POS_FRAMES,left_pos)
#        vid_middle.set(cv2.CAP_PROP_POS_FRAMES,middle_pos)
#        vid_right.set(cv2.CAP_PROP_POS_FRAMES,right_pos)
#        ret,img_l = vid_left.read()
#        ret,img_m = vid_middle.read()
#        ret,img_r = vid_right.read()
#    
#        # Grab radiance image of depth cameras
#        img_e40 = vid_e40.read(e40_pos)
#        img_e70 = vid_e70.read(e70_pos)
#        # Normalize and convert to grayscale image
#        img_e40 = (255*((img_e40-img_e40.min())/(img_e40.max()-img_e40.min()))).astype(np.uint8)
#        img_e70 = (255*((img_e70-img_e70.min())/(img_e70.max()-img_e70.min()))).astype(np.uint8)
#        # Resize to facilitate chessboard detection (see factor value in preamble)
#        img_e40_rs = cv2.resize(img_e40,(int(depthcam_scalefactor*img_e40.shape[1]),int(depthcam_scalefactor*img_e40.shape[0])),0,0,cv2.INTER_LINEAR )
#        img_e70_rs = cv2.resize(img_e70,(int(depthcam_scalefactor*img_e70.shape[1]),int(depthcam_scalefactor*img_e70.shape[0])),0,0,cv2.INTER_LINEAR )
#    
#        # Look for the chessboard corners in each image
#        ret_l, corners_l = cv2.findChessboardCorners(img_l, (9,6), flags=chess_detection_flags )
#        if ret_l:
#            cv2.imwrite(os.path.join(calib_path,'%02d_left_SEQ%d_fr%04d.png'%(calib_iter,4,left_pos)),img_l)
#        
#        ret_m, corners_m = cv2.findChessboardCorners(img_m, (9,6), flags=chess_detection_flags )
#        if ret_m: 
#            cv2.imwrite(os.path.join(calib_path,'%02d_middle_SEQ%d_fr%04d.png'%(calib_iter,4,middle_pos)),img_m)
#            
#        ret_r, corners_r = cv2.findChessboardCorners(img_r, (9,6), flags=chess_detection_flags )
#        if ret_r: 
#            cv2.imwrite(os.path.join(calib_path,'%02d_right_SEQ%d_fr%04d.png'%(calib_iter,4,right_pos)),img_r)
#            
#        ret_e70, corners_e70 = cv2.findChessboardCorners(img_e70_rs, (9,6), flags=chess_detection_flags )
#        if ret_e70: 
#            cv2.imwrite(os.path.join(calib_path,'%02d_e70_SEQ%d_fr%04d.png'%(calib_iter,4,e70_pos)),img_e70)
#            
#        ret_e40, corners_e40 = cv2.findChessboardCorners(img_e40_rs, (9,6), flags=chess_detection_flags )
#        if ret_e40: 
#            cv2.imwrite(os.path.join(calib_path,'%02d_e40_SEQ%d_fr%04d.png'%(calib_iter,4,e40_pos)),img_e40)        
#    
#        # If chessboard has been detected on at least one view
#        if ret_l or ret_m or ret_r or ret_e40 or ret_e70:
#            # increase iterator
#            calib_iter += 1    
#        
#        # Moving to the next image
#        left_pos += step
#    
#    # Close window
#    cv2.destroyAllWindows()
#    
#    # Release video files
#    vid_left.release()
#    vid_middle.release()
#    vid_right.release()
#    vid_e70.close()
#    vid_e40.close()
#    
#
#    #__________________________________________________________________________    
#    # SEQUENCE #5
#    # color cams
#    cam_left = datapath+'Left/SEQ5_LEFT_MVI_0394.MOV'
#    cam_middle = datapath+'Middle/SEQ5_MIDDLE_MVI_0004.MOV'
#    cam_right = datapath+'Right/SEQ5_RIGHT_MVI_0568.MOV'
#    middle_offset = int(+27)
#    right_offset = int(+62)
#    left_time_calibstart = 34
#    left_time_calibend = 51
#    # depth cams (but radiance sequences)
#    cam70_b = datapath+'E70/SEQ5_E70_FZ_2014-05-23_11333301_b.pgm'
#    cam40_b = datapath+'E40/SEQ5_E40_FZ_2014-05-23_11333601_b.pgm'
#    cam70_offset = -100
#    cam70_factor = 0.963
#    cam40_offset = -180
#    cam40_factor = 0.963
#    
#
#    # Initializing the 5 video files
#    vid_left = cv2.VideoCapture(cam_left)
#    if(not(vid_left.isOpened())):
#        print("Left file cannot be opened")
#        exit(0)
#    
#    vid_middle = cv2.VideoCapture(cam_middle)
#    if(not(vid_middle.isOpened())):
#        print("Middle file cannot be opened")
#        exit(0)
#        
#    vid_right = cv2.VideoCapture(cam_right)
#    if(not(vid_right.isOpened())):
#        print("Right file cannot be opened")
#        exit(0)
#    
#    vid_e70 = readPGM(cam70_b)
#    vid_e70.open()
#    vid_e40 = readPGM(cam40_b)
#    vid_e40.open()
#    
#    # Displaying media information
#    nbf_left, fps_left, w_left, h_left = vid_left.get(cv2.CAP_PROP_FRAME_COUNT), vid_left.get(cv2.CAP_PROP_FPS), vid_left.get(cv2.CAP_PROP_FRAME_WIDTH), vid_left.get(cv2.CAP_PROP_FRAME_HEIGHT)
#    print('Left video:',w_left,'x',h_left,'-',nbf_left,'frames @', fps_left,'fps.')
#    nbf_middle, fps_middle, w_middle, h_middle = vid_middle.get(cv2.CAP_PROP_FRAME_COUNT), vid_middle.get(cv2.CAP_PROP_FPS), vid_middle.get(cv2.CAP_PROP_FRAME_WIDTH), vid_middle.get(cv2.CAP_PROP_FRAME_HEIGHT)
#    print('Middle video:',w_middle,'x',h_middle,'-',nbf_middle,'frames @', fps_middle,'fps.')
#    nbf_right, fps_right, w_right, h_right = vid_right.get(cv2.CAP_PROP_FRAME_COUNT), vid_right.get(cv2.CAP_PROP_FPS), vid_right.get(cv2.CAP_PROP_FRAME_WIDTH), vid_right.get(cv2.CAP_PROP_FRAME_HEIGHT)
#    print('Right video:',w_right,'x',h_right,'-',nbf_right,'frames @', fps_right,'fps.')
#    
#    print('Depth video E70:',vid_e70.width,'x',vid_e70.height,'-',vid_e70.nbframes,'frames @', fps_left*cam70_factor,'fps.')
#    print('Depth video E40:',vid_e40.width,'x',vid_e40.height,'-',vid_e40.nbframes,'frames @', fps_left*cam40_factor,'fps.')
#    
#    
#    # Chessboard detection
#    # Definition of searching range
#    left_pos_start = int(left_time_calibstart*fps_left)
#    left_pos_end = int(left_time_calibend*fps_left)
#    
#
#    left_pos = left_pos_start
#    while left_pos<left_pos_end:    # left cam index is used as reference
#    
#        # Adjusting index in each other video
#        middle_pos = left_pos+middle_offset
#        right_pos = left_pos+right_offset
#        e70_pos = int(np.round(cam70_factor*left_pos+cam70_offset))
#        e40_pos = int(np.round(cam40_factor*left_pos+cam40_offset))
#        print(left_pos,middle_pos,right_pos,e40_pos,e70_pos)
#    
#        # Move to index and grab the frames
#        vid_left.set(cv2.CAP_PROP_POS_FRAMES,left_pos)
#        vid_middle.set(cv2.CAP_PROP_POS_FRAMES,middle_pos)
#        vid_right.set(cv2.CAP_PROP_POS_FRAMES,right_pos)
#        ret,img_l = vid_left.read()
#        ret,img_m = vid_middle.read()
#        ret,img_r = vid_right.read()
#    
#        # Grab radiance image of depth cameras
#        img_e40 = vid_e40.read(e40_pos)
#        img_e70 = vid_e70.read(e70_pos)
#        # Normalize and convert to grayscale image
#        img_e40 = (255*((img_e40-img_e40.min())/(img_e40.max()-img_e40.min()))).astype(np.uint8)
#        img_e70 = (255*((img_e70-img_e70.min())/(img_e70.max()-img_e70.min()))).astype(np.uint8)
#        # Resize to facilitate chessboard detection (see factor value in preamble)
#        img_e40_rs = cv2.resize(img_e40,(int(depthcam_scalefactor*img_e40.shape[1]),int(depthcam_scalefactor*img_e40.shape[0])),0,0,cv2.INTER_LINEAR )
#        img_e70_rs = cv2.resize(img_e70,(int(depthcam_scalefactor*img_e70.shape[1]),int(depthcam_scalefactor*img_e70.shape[0])),0,0,cv2.INTER_LINEAR )
#    
#        # Look for the chessboard corners in each image
#        ret_l, corners_l = cv2.findChessboardCorners(img_l, (9,6), flags=chess_detection_flags )
#        if ret_l:
#            cv2.imwrite(os.path.join(calib_path,'%02d_left_SEQ%d_fr%04d.png'%(calib_iter,5,left_pos)),img_l)
#        
#        ret_m, corners_m = cv2.findChessboardCorners(img_m, (9,6), flags=chess_detection_flags )
#        if ret_m: 
#            cv2.imwrite(os.path.join(calib_path,'%02d_middle_SEQ%d_fr%04d.png'%(calib_iter,5,middle_pos)),img_m)
#            
#        ret_r, corners_r = cv2.findChessboardCorners(img_r, (9,6), flags=chess_detection_flags )
#        if ret_r: 
#            cv2.imwrite(os.path.join(calib_path,'%02d_right_SEQ%d_fr%04d.png'%(calib_iter,5,right_pos)),img_r)
#            
#        ret_e70, corners_e70 = cv2.findChessboardCorners(img_e70_rs, (9,6), flags=chess_detection_flags )
#        if ret_e70: 
#            cv2.imwrite(os.path.join(calib_path,'%02d_e70_SEQ%d_fr%04d.png'%(calib_iter,5,e70_pos)),img_e70)
#            
#        ret_e40, corners_e40 = cv2.findChessboardCorners(img_e40_rs, (9,6), flags=chess_detection_flags )
#        if ret_e40: 
#            cv2.imwrite(os.path.join(calib_path,'%02d_e40_SEQ%d_fr%04d.png'%(calib_iter,5,e40_pos)),img_e40)         
#    
#        # If chessboard has been detected on at least one view
#        if ret_l or ret_m or ret_r or ret_e40 or ret_e70:
#            # increase iterator
#            calib_iter += 1    
#        
#        # Moving to the next image
#        left_pos += step
#    
#    # Close window
#    cv2.destroyAllWindows()
#    
#    # Release video files
#    vid_left.release()
#    vid_middle.release()
#    vid_right.release()
#    vid_e70.close()
#    vid_e40.close()
            