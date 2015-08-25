
import cv2
import numpy as np
import os,sys
import math as m
import pandas as pd

def showLinkedTracks(filename, start, frames, outputfilename):
    

    linkedDF = pd.read_csv('../classify/output.csv') 
    
    
    

    
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
 
    S = (1920,1080)
    out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 2, S, True)
    
    for tt in range(frames):
        
        # Capture frame-by-frame
        _, frame = cap.read()
        if (tt%15) > 0 : continue
        thisFrame = linkedDF.ix[linkedDF['frame']==tt]

        
        # draw detected objects and display
        sz=6
        
        for i, row in thisFrame.iterrows():
            #if int(row['particle'])!=628:
            #    continue
            
            #cv2.putText(frame ,str(int(row['particle'])) ,((int(row['x'])+12, int(row['y'])+12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,255,2)
            cv2.rectangle(frame, ((int( row['x'])-sz, int( row['y'])-sz)),((int( row['x'])+sz, int( row['y'])+sz)),(0,0,0),2)
            
        cv2.imshow('frame',frame)
        out.write(frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    

    cv2.destroyAllWindows()
    cap.release()
    out.release()

if __name__ == '__main__':
    FULLNAME = sys.argv[1]
    frameStart = 0
    frameLength = int(sys.argv[3])
    path, filename = os.path.split(FULLNAME)
    noext, ext = os.path.splitext(filename)
    allTransforms=np.zeros((frameLength,3))
    outputfilename = noext + '.csv' 
    showLinkedTracks(FULLNAME, frameStart, frameLength, outputfilename)
