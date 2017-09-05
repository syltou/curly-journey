# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:51:15 2017

@author: sylvto
"""


import numpy as np
import cv2
import math
import datetime
from imgTools import show
from scipy.interpolate import interp2d



def eulerAnglesToRotationMatrix(theta) :
    ''' Return rotation matrix from three Euler angles (in order X, Y, Z) '''

    theta = np.array(theta)/180. * math.pi

    if(theta[0]==0):
        R_x = np.eye(3)
    else:
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                        [0,         math.sin(theta[0]), math.cos(theta[0])  ]])
         
    if(theta[1]==0):
        R_y = np.eye(3)
    else:                 
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]])
    
    if(theta[2]==0):
        R_z = np.eye(3)
    else:             
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]])
                     
                     
    R = np.dot(R_x, np.dot( R_y, R_z ))
 
    return R



def distortPoints(pts, distCoeffs):
    
    k1 = distCoeffs[0,0]
    k2 = distCoeffs[0,1]
    p1 = distCoeffs[0,2]
    p2 = distCoeffs[0,3]
    k3 = distCoeffs[0,4]

    newpts = []
    for pt in pts:
        
        x = pt[0]
        y = pt[1]
        
        rs = x*x + y*y
        
        # radial correction
        newx = x * (1 + k1*rs + k2*rs*rs + k3*rs*rs)
        newy = y * (1 + k1*rs + k2*rs*rs + k3*rs*rs)
        
        # tangential correction
        newx = newx + (2*p1*x*y + p2*(rs + 2*x*x))
        newy = newy + (p1*(rs + 2*y*y) + 2*p2*x*y)
        
        newpt = [newx,newy]
        
        newpts.append(newpt)

    
    return newpts


def undistortImage(img, cameraMatrix, distCoeffs, crop=False):
    
    h,  w = img.shape[:2]
    new_cameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h), 1, (w,h))
    
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, new_cameraMatrix, (w,h), cv2.CV_32FC1 )
    new_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    
    if(crop): # crop the image
        x, y, w, h = roi
        new_img = new_img[y:y+h, x:x+w]
    
    return new_img



def undistortPoints( points, cameraMatrix, distCoeffs, imgsize):
    
    new_cameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imgsize, 1, imgsize)
    
    # undistort
    new_points = cv2.undistortPoints(points, cameraMatrix, distCoeffs, P=new_cameraMatrix )
    
    return new_points



def reprojectToCalibrationImage(cameraMatrix, distCoeffs, rvecs, tvecs, objpoints, imgpoints):
    
    avg_errors = []
    for k in range(len(rvecs)):
        rvec = rvecs[k]
        tvec = tvecs[k]
        R,_ = cv2.Rodrigues(rvec)
        
        errors = []
        for i,(objpt,imgpt) in enumerate(zip(objpoints[k],imgpoints[k])):
            
            # project model point to world coordinates
            worldpt = np.dot(R,objpt.reshape(objpt.size,1))+tvec
            # apply lens distortions
            distorted_worldpt = distortPoints([[worldpt[0,0]/worldpt[2,0],worldpt[1,0]/worldpt[2,0]]],distCoeffs)
            # reshape world coordinates to homogeneous coordinates
            worldpt[0,0] = distorted_worldpt[0,0]
            worldpt[1,0] = distorted_worldpt[0,1]
            worldpt[2,0] = 1.0
            # project to image coordinates using camera matrix            
            new_imgpt = np.dot(cameraMatrix,worldpt)
            # reshape
            new_imgpt = (new_imgpt.reshape(1,new_imgpt.size))[:,:2]

            ### The seven lines above are exactly equivalent to the next one (if distortion coefficients do not contain k4, k5 and k6)
            #new_imgpt2,_ = cv2.projectPoints(np.array([objpt],dtype='float64'),rvec,tvec,cameraMatrix,distCoeffs)
            
            # compute the distance between original image point and its reprojection
            distance = np.linalg.norm(imgpt-new_imgpt)
            # store the point error as the squared distance
            errors.append(distance*distance)

        # for each image, store the average squared distance between original image point and reprojected image points
        avg_errors.append(np.average(errors))


    # index of the calibration image with the biggest average error 
    index = np.argmax(avg_errors)
    # biggest error, used to reject calibration image if above threshold
    max_error = avg_errors[index]
    # global average error is computed as the squareroot of the average squared distance to be similar to the value reported by cv2.calibrateCamera
    avg_error = np.sqrt(np.average(avg_errors))
        
    return index, max_error, avg_error



def depthMapping(depth,K_depth,R,T,K_rgb,rgbsize,undistortMap=None,distCoeffs_depth=None):
    ''' Project depth value unto RGB image coordinates '''
    
    (h_depth,w_depth) = depth.shape
    (h_rgb,w_rgb) = rgbsize
    
    new_depth = depth.max()*np.ones(rgbsize)
    
#    print(K_depth)
#    print(R)
#    print(T)
#    print(K_rgb)
    
    if(undistortMap is not None):
        if(distCoeffs_depth is None):
            print('Cannot undistort if no distortion coefficients are passed')
        else:
            depth = undistortImage(depth, K_depth, distCoeffs_depth, crop=False)
    
    
    for x in range(w_depth):
        for y in range(h_depth):
            
            Z = depth[y,x]
            
            if(Z!=0):
                
                # 3D coordinates
                X = (x - K_depth[0,2]) * Z / K_depth[0,0]
                Y = (y - K_depth[1,2]) * Z / K_depth[1,1]
                Point3D_depth = np.array([X,Y,Z])
                
#                print(x,y,X,Y,Z)
                
                # reproject
                Point3D_rgb = np.dot(R,Point3D_depth[:,np.newaxis]) + T
    
                x_rgb = Point3D_rgb[0] * K_rgb[0,0] / Point3D_rgb[2] + K_rgb[0,2]
                y_rgb = Point3D_rgb[1] * K_rgb[1,1] / Point3D_rgb[2] + K_rgb[1,2]
                
#                print(Point3D_rgb,x_rgb,y_rgb)
                
                
                
    #            print(int(y_rgb),int(x_rgb))
                
                if(x_rgb>=0 and x_rgb<w_rgb and y_rgb>=0 and y_rgb<h_rgb):
                    if(depth[y,x]<new_depth[int(y_rgb),int(x_rgb)]):
                        new_depth[int(y_rgb),int(x_rgb)] = depth[y,x]
    
    return new_depth


def viewRendering(view,depth,K,new_K,R,T,undistortMap=None,distCoeffs_depth=None):
    ''' Project depth value unto RGB image coordinates '''
    
    (h,w) = view.shape[:2]
    new_view = np.ones(view.shape)
     
    if(undistortMap is not None):
        if(distCoeffs_depth is None):
            print('Cannot undistort if no distortion coefficients are passed')
        else:
            depth = undistortImage(depth, K, distCoeffs_depth, crop=False)
            view = undistortImage(view, K, distCoeffs_depth, crop=False)
    
    for u in range(w):
        for v in range(h):
            
            Z = depth[v,u]
            
            if(Z!=0):
                
                # 3D coordinates
                X = (u - K[0,2]) * Z / K[0,0]
                Y = (v - K[1,2]) * Z / K[1,1]
                Point3D = np.array([X,Y,Z])
                
                # reproject
                new_Point3D = np.dot(R,Point3D[:,np.newaxis]) + T
    
                new_x = new_Point3D[0] * new_K[0,0] / new_Point3D[2] + new_K[0,2]
                new_y = new_Point3D[1] * new_K[1,1] / new_Point3D[2] + new_K[1,2]
                
                new_x = int(np.round(new_x))
                new_y = int(np.round(new_y))
                
                if(new_x>=0 and new_x<w and new_y>=0 and new_y<h):
                    new_view[new_y,new_x] = view[v,u]
    
    return new_view


def viewRendering2(view,depth,K,new_K,R,T,undistortMap=None,distCoeffs_depth=None):
    ''' Project depth value unto RGB image coordinates '''
    
    (h,w) = view.shape[:2]
    new_view = np.zeros(view.shape)
    
    if(undistortMap is not None):
        if(distCoeffs_depth is None):
            print('Cannot undistort if no distortion coefficients are passed')
        else:
            depth = undistortImage(depth, K, distCoeffs_depth, crop=False)
            view = undistortImage(view, K, distCoeffs_depth, crop=False)
    
    # src image sensor coordinates
    [x,y] = np.meshgrid(range(w),range(h))
    
    # world coordinates
    X = (x - K[0,2]) * depth / K[0,0]
    Y = (y - K[1,2]) * depth / K[1,1]
    Z = depth
    
    Points3D = np.array( [ np.reshape(X,X.size),
                           np.reshape(Y,Y.size),
                           np.reshape(Z,Z.size)]  )
    
    # projection onto the new camera plane
    new_Points3D = np.dot(R,Points3D) + T
    
#    new_X = np.reshape(new_Points3D[0],(h,w))
#    new_Y = np.reshape(new_Points3D[1],(h,w))
#    new_Z = np.reshape(new_Points3D[2],(h,w))
    new_X = new_Points3D[0]
    new_Y = new_Points3D[1]
    new_Z = new_Points3D[2]
    
    # conversion to sensor coordinates
    new_x = new_X * new_K[0,0] / new_Z + new_K[0,2]
    new_y = new_Y * new_K[1,1] / new_Z + new_K[1,2]
    
    print(new_x.shape)
    
    filt = np.logical_and( np.logical_and( new_x>=-.5, new_x<w-.5), np.logical_and( new_y>=-.5, new_y<h-.5))
    new_x = np.round(new_x[filt]).astype('int')
    new_y = np.round(new_y[filt]).astype('int')
    
    x = x.ravel()[filt]
    y = y.ravel()[filt]

    new_view[new_y,new_x] = view[y,x]
    

    return new_view


def projectForward(colour_src,depth_src,K_src,K_dst,R,T,undistortMap=None,distCoeffs_src=None):
    ''' Project view and depth images to new camera 
        - colour_src and depth_src are resp. the rgb texture and the depth map of the src scene 
        - K_src is intrinsic matrix of src camera
        - K_dst is intrinsic matrix of dst camera
        - R and T are rotation matrix and translation vector between src and dst '''
    
    colour_dst = np.zeros(colour_src.shape)
    depth_dst = np.zeros(depth_src.shape)
    (h,w) = colour_src.shape[:2]
    
    if(undistortMap is not None):
        if(distCoeffs_src is None):
            print('Cannot undistort if no distortion coefficients are passed')
        else:
            depth_src = undistortImage(depth_src, K_src, distCoeffs_src, crop=False)
            colour_src = undistortImage(colour_src, K_src, distCoeffs_src, crop=False)
    
    # src image sensor coordinates
    [x_src,y_src] = np.meshgrid(range(w),range(h))
    
    # projection to world coordinates
    X = (x_src - K_src[0,2]) * depth_src / K_src[0,0]
    Y = (y_src - K_src[1,2]) * depth_src / K_src[1,1]
    Z = depth_src
    
    Points3D_src = np.array( [    np.reshape(X,X.size),
                                  np.reshape(Y,Y.size),
                                  np.reshape(Z,Z.size)  ])
    
    # Rotation and translation to new pose
    Points3D_dst = np.dot(R,Points3D_src) + T
    
    # conversion to sensor coordinates
    x_dst = Points3D_dst[0] * K_dst[0,0] / Points3D_dst[2] + K_dst[0,2]
    y_dst = Points3D_dst[1] * K_dst[1,1] / Points3D_dst[2] + K_dst[1,2]
    
    # create a mask with only position inside the dst image
    filt = np.logical_and( np.logical_and( x_dst>=-.5, x_dst<w-.5), np.logical_and( y_dst>=-.5, y_dst<h-.5))
    # apply mask on both the original set of coordinates and their corresponding projections
    x_src = x_src.ravel()[filt]
    y_src = y_src.ravel()[filt]
    x_dst = np.round(x_dst[filt]).astype('int')
    y_dst = np.round(y_dst[filt]).astype('int')

    # map the src image to the dst image
    colour_dst[y_dst,x_dst] = colour_src[y_src,x_src]
    depth_dst[y_dst,x_dst] = depth_src[y_src,x_src]
    

    return colour_dst, depth_dst



def projectBackward(colour_src,depth_dst,K_src,K_dst,R,T,undistortMap=None,distCoeffs_src=None):
    ''' Project view and depth images to new camera 
        - colour_src and depth_src are resp. the rgb texture and the depth map of the src scene 
        - K_src is intrinsic matrix of src camera
        - K_dst is intrinsic matrix of dst camera
        - R and T are rotation matrix and translation vector between src and dst '''
    
    colour_dst = np.zeros(colour_src.shape)
    (h,w) = colour_src.shape[:2]
    
    if(undistortMap is not None):
        if(distCoeffs_src is None):
            print('Cannot undistort if no distortion coefficients are passed')
        else:
            colour_src = undistortImage(colour_src, K_src, distCoeffs_src, crop=False)
    
    # dst image sensor coordinates
    [x_dst,y_dst] = np.meshgrid(range(w),range(h))
    
    # projection to world coordinates
    X = (x_dst - K_dst[0,2]) * depth_dst / K_dst[0,0]
    Y = (y_dst - K_dst[1,2]) * depth_dst / K_dst[1,1]
    Z = depth_dst
    
    Points3D_dst = np.array( [    np.reshape(X,X.size),
                                  np.reshape(Y,Y.size),
                                  np.reshape(Z,Z.size)  ])
    
    
    
    # Rotation and translation to new pose
    Points3D_src = np.dot(R.T,Points3D_dst) - T
    
    # conversion to sensor coordinates
    x_src = Points3D_src[0] * K_src[0,0] / Points3D_src[2] + K_src[0,2]
    y_src = Points3D_src[1] * K_src[1,1] / Points3D_src[2] + K_src[1,2]
    
    # create a mask with only position inside the dst image
    filt = np.logical_and( np.logical_and( x_src>=-.5, x_src<w-.5), np.logical_and( y_src>=-.5, y_src<h-.5))
    # apply mask on both the original set of coordinates and their corresponding projections
    x_dst = x_dst.ravel()[filt]
    y_dst = y_dst.ravel()[filt]
    x_src = np.round(x_src[filt]).astype('int')
    y_src = np.round(y_src[filt]).astype('int')

    # map the src image to the dst image
    colour_dst[y_dst,x_dst] = colour_src[y_src,x_src]
    

    return colour_dst




def viewRenderingFast(src,dst,depth,K,new_K,R,T,undistortMap=None,distCoeffs=None,overwrite=False):
    ''' Project depth value unto RGB image coordinates '''
    
    src = np.double(src)
    depth = np.double(depth)
    
    (h,w) = src.shape[:2]
    
#    print(K_depth)
#    print(R)
#    print(T)
#    print(K_rgb)
    
    if(undistortMap is not None):
        if(distCoeffs is None):
            print('Cannot undistort if no distortion coefficients are passed')
        else:
            depth = undistortImage(depth, K, distCoeffs, crop=False)
            src = undistortImage(src, K, distCoeffs, crop=False)
    
    # src image sensor coordinates
    [x,y] = np.meshgrid(range(w),range(h))
    
    # world coordinates
    X = (x - K[0,2]) * depth / K[0,0]
    Y = (y - K[1,2]) * depth / K[1,1]
    Z = depth
    
    Points3D = np.array( [ np.reshape(X,X.size),
                           np.reshape(Y,Y.size),
                           np.reshape(Z,Z.size)]  )
    
    # projection onto the new camera plane
    new_Points3D = np.dot(R,Points3D) + T
    
    new_X = np.reshape(new_Points3D[0],(h,w))
    new_Y = np.reshape(new_Points3D[1],(h,w))
    new_Z = np.reshape(new_Points3D[2],(h,w))
    
    
    # conversion to sensor coordinates
    new_x = new_X * new_K[0,0] / new_Z + new_K[0,2]
    new_y = new_Y * new_K[1,1] / new_Z + new_K[1,2]
    
    
    (new_x,new_y) = cv2.convertMaps((new_x.astype('float32')), (new_y.astype('float32')), dstmap1type=cv2.CV_32FC1)
    temp = cv2.remap(src,new_x,new_y,cv2.INTER_LINEAR)
    
    if overwrite:
        dst[:,:,:] = temp[:,:,:]



