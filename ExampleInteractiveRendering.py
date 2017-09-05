# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:12:34 2017

@author: sylvto
"""

import cv2
#import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from Geometry import viewRendering, viewRenderingFast, eulerAnglesToRotationMatrix
from imgTools import show

import math
import datetime




cv2.destroyAllWindows()

## Original camera parameters
#############################
# Intrinsic parameters of original camera
K_original = np.array( [[1732.87,    0.0,        943.23    ],
                        [0.0,        1729.90,    548.845040],
                        [0,          0,          1         ]])

# Extrinsic parameters of original camera
Rt_original = np.array([    [1.0,    0.0,    0.0,    0.0],
                            [0.0,    1.0,    0.0,    0.0],
                            [0.0,    0.0,    1.0,    0.0]])

# Znear and Zfar are nearest and fartheset points in the scene from the original camera
Zfar = 2760.510889
Znear = 34.506386

## Virtual camera parameters
#############################
# Intrinsic parameters of virtual camera
K_virtual = np.array([  [1732.87,    0.0,        943.23],
                        [0.0,        1729.90,    548.845040],
                        [0,          0,          1]])

# Extrinsic parameters of virtual camera
R0 = np.eye(3)
T0 = np.zeros((3,1))

R = np.eye(3)
T = np.zeros((3,1))




scale = .5




D = cv2.imread(u"..//..//Task//D_original.png")
D = D[:,:,0]
V = cv2.imread(u"..//..//Task//V_original.png")

D = cv2.resize(D,None,fx=scale, fy=scale, interpolation = cv2.INTER_LINEAR)
V = cv2.resize(V,None,fx=scale, fy=scale, interpolation = cv2.INTER_LINEAR)

for (i,j) in [(0,0),(0,2),(1,1),(1,2)]:
    K_original[i,j] *= scale
    K_virtual[i,j] *= scale

V = np.double(V)
D = np.double(D)
D = Znear + (Zfar - Znear)*(D-np.min(D))/(np.max(D)-np.min(D))

(h,w) = D.shape



drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1


# mouse callback function
def render(event,x,y,flags,param):
    global ix,iy,drawing,mode,V,w,h,R,R0,T,T0

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            dx = x-ix
            dy = y-iy
            angley = -dx/w*math.pi/5.
            anglex = dy/w*math.pi/5.
            R = eulerAnglesToRotationMatrix([anglex,angley,0])            
            viewRenderingFast(V,img,D,K_original,K_virtual,np.dot(R0,R),T0+T,overwrite=True)
            

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False        
        viewRenderingFast(V,img,D,K_original,K_virtual,np.dot(R0,R),T0+T,overwrite=True)
        R0 = R
        
        
        




## Create a black image, a window and bind the function to window
#img = np.zeros((512,512,3), np.uint8)
#cv2.namedWindow('image')
#cv2.setMouseCallback('image',draw_circle)
#
#while(1):
#    cv2.imshow('image',img)
#    if cv2.waitKey(20) & 0xFF == 27:
#        break
#cv2.destroyAllWindows()




#    start = datetime.datetime.now()
#    V1 = viewRendering(V,D,K_original,K_virtual,R,T)
#    print('Method 1:',datetime.datetime.now()-start)



img = np.zeros(V.shape)


#viewRenderingFast(V,img,D,K_original,K_virtual,R,T,overwrite=True)
#show(V,'original')
#show(img.astype('uint8'),'warped')
#cv2.waitKey(0)
#cv2.destroyAllWindows()



    
cv2.namedWindow('image')
cv2.setMouseCallback('image',render)

while(1):
    cv2.imshow('image',img.astype('uint8'))
    k = cv2.waitKey(1) & 0xFF
    if k == ord('d'):
        T[0,0] += 1 
        viewRenderingFast(V,img,D,K_original,K_virtual,np.dot(R0,R),T0+T,overwrite=True)
    if k == ord('a'):
        T[0,0] -= 1 
        viewRenderingFast(V,img,D,K_original,K_virtual,np.dot(R0,R),T0+T,overwrite=True)
    if k == ord('w'):
        T[1,0] += 1 
        viewRenderingFast(V,img,D,K_original,K_virtual,np.dot(R0,R),T0+T,overwrite=True)
    if k == ord('s'):
        T[1,0] -= 1 
        viewRenderingFast(V,img,D,K_original,K_virtual,np.dot(R0,R),T0+T,overwrite=True)
    elif k == 27:
        break
cv2.destroyAllWindows()

