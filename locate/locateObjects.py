
import cv2
import numpy as np
import os,sys
import math as m
import pandas as pd

def locate(filename, outputfilename):
    

    df = pd.DataFrame(columns= ['x', 'y', 'mass', 'size', 'ecc', 'signal', 'ep', 'frame'])
    
    

    
    cap = cv2.VideoCapture(filename)
     
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)  )
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
        print(tt)
        # Capture frame-by-frame
        _, frame = cap.read()
        if (tt%15) > 0 : continue
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2image = 255-cv2image
 
        blobs= blobdetector.detect(cv2image)
        # draw detected objects and display
        sz=6
        thisFrame = pd.DataFrame(columns= ['x', 'y', 'mass', 'size', 'ecc', 'signal', 'ep', 'frame'])
        ind = 0
        for b in blobs:
            ind +=1
            cv2.rectangle(frame, ((int(b.pt[0])-sz, int(b.pt[1])-sz)),((int(b.pt[0])+sz, int(b.pt[1])+sz)),(0,0,0),2)
            thisFrame.set_value(ind, 'x', b.pt[0])
            thisFrame.set_value(ind, 'y', b.pt[1])
            thisFrame.set_value(ind, 'frame', tt)
        df = pd.concat([df,thisFrame])

        #cv2.imshow('frame',frame)
        #k = cv2.waitKey(30) & 0xff
        #if k == 27:
        #    break
    
    df.to_csv(outputfilename)
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    FULLNAME = sys.argv[1]
    path, filename = os.path.split(FULLNAME)
    noext, ext = os.path.splitext(filename)
    outputfilename = noext + '.csv' 
    locate(FULLNAME,  outputfilename)
