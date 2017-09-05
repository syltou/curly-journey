# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:23:00 2017

@author: Sylvain
"""

import cv2
#import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from Geometry import viewRendering, viewRendering2, viewRenderingFast, eulerAnglesToRotationMatrix, projectForward, projectBackward
from imgTools import show

import math
import datetime
import matplotlib




#cv2.destroyAllWindows()

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
T[0,0]=5


R = eulerAnglesToRotationMatrix([0,1.5,0])

angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)

print(angles)
print(R)

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

show(D,'depth')
show(V,'texture')

#start = datetime.datetime.now()
#D2 = viewRendering(V,D,K_original,K_virtual,R,T)
#print(datetime.datetime.now()-start)
#show(D2,'toto')

start = datetime.datetime.now()
V2,D2 = projectForward(V,D,K_original,K_virtual,R,T)
D3    = projectBackward(D,D2,K_original,K_virtual,R,T)
print(datetime.datetime.now()-start)
show(V2,'forward project')
show(D2,'forward project')
show(D3,'backward project')

#show(np.log10(1+np.abs(V2-V3)),'diff')

cv2.waitKey(0)
cv2.destroyAllWindows()
