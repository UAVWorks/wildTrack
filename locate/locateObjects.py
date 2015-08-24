
import cv2
import numpy as np
import os,sys
import math as m
import pandas as pd

def locate(filename, start, frames, outputfilename):
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
 
    # parameters for object detection    
    # find out more about parameters here http://www.learnopencv.com/blob-detection-using-opencv-python-c/
    params = cv2.SimpleBlobDetector_Params()
    params.maxThreshold= 100
    params.minThreshold= 25
    params.thresholdStep= 1
    params.filterByArea= 1
    params.maxArea= 25
    params.minArea= 4
    params.filterByCircularity= 0
    params.filterByInertia= 0
    params.filterByConvexity= 0


    blobdetector = cv2.SimpleBlobDetector_create(params)

    
    for tt in range(frames):
        # Capture frame-by-frame
        _, frame = cap.read()
        
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2image = 255-cv2image
 
        blobs= blobdetector.detect(cv2image)
        # draw detected objects and display
        sz=6
        for b in blobs:
            cv2.rectangle(frame, ((int(b.pt[0])-sz, int(b.pt[1])-sz)),((int(b.pt[0])+sz, int(b.pt[1])+sz)),(0,0,0),2)
        cv2.imshow('framea',frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    FULLNAME = sys.argv[1]
    frameStart = 0
    frameLength = int(sys.argv[3])
    path, filename = os.path.split(FULLNAME)
    noext, ext = os.path.splitext(filename)
    allTransforms=np.zeros((frameLength,3))
    outputname = noext + '_FRW' + str(frameStart) + '.avi' 
    locate(FULLNAME, frameStart, frameLength, outputname)
