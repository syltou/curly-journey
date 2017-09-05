from Calibration import predetectChessboardAuto,getCalibrationImages,calibrateMonoCamera,calibrateStereoCamera
import cv2
import os
import numpy as np
from imgTools import show
from Geometry import undistortImage

def main():
    
    calib_path = u'.//CalibAuto'

    cam1 = {'name':'left','id':2}
    cam2 = {'name':'right','id':0}
    cams = [cam1,cam2]
    
    
    ### Automatic detection of chessboards
    predetectChessboardAuto(cams,calib_path)


    ### Individual calibration of each camera
    for cam in cams:

        archive = os.path.join(calib_path,'calib_'+cam['name']+'.npz')
        
        # listing calibration images concerning selected camera
        calibimages = getCalibrationImages(calib_path,cam['name'])
        # individual calibration from the calibration images
        cameraMatrix, distCoeffs, imgpoints, calibrationimages = calibrateMonoCamera(calibimages, visualize = True)
        print(cameraMatrix)
        print(distCoeffs)
        
        # saving results
        np.savez(archive, K=cameraMatrix, k=distCoeffs, imgpoints=imgpoints, calibrationimages=calibrationimages)
        
        
        # check distortion correction
        vid = cv2.VideoCapture(cam['id'])
        
        while(1):        
            
            ret,img = vid.read()
            img_corrected = undistortImage(img, cameraMatrix, distCoeffs, crop=False)
            img_disp = np.concatenate((img,img_corrected),1)
            
            show(img_disp,'Checking the correction of distortion. Left: Original, Right: Corrected',stats=False)  
            key = cv2.waitKey(10)
            if key==27:   # if ESC is pressed, abort
                break
         
        cv2.destroyAllWindows()

    
    ### Stereo calibration    
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = calibrateStereoCamera(cam1['name'],cam2['name'],calib_path=calib_path)
    
    print('Rotation matrix')
    print(R)
    print('Translation vector')
    print(T)
    # Euler angles
    ret,_,_,_,_,_ = cv2.RQDecomp3x3(R)
    print('Rotation angles:',ret)
    
    
    
    
if __name__ == "__main__":
    main()

