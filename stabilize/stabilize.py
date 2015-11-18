import cv2
import numpy as np
import os,sys
import math as m

def processMovie(filename, start, frames, outputfilename):
    reduceFPS=4
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    S = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(outputfilename, cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS)/reduceFPS, S, True)
 

    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 200;
     
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-4;
     
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
     

    im1_gray = np.array([])
    first = np.array([])
    for tt in range(frames):
        # Capture frame-by-frame
        _, frame = cap.read()
        if not(im1_gray.size):
            im1_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            first = frame.copy()
        
        im2_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        try:
            (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
        except cv2.error as e:
            im1_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            first = frame.copy()
            print("missed frame")
        
        im2_aligned = np.empty_like(frame)
        np.copyto(im2_aligned, first)
        # transform the frame - the 5s are to cut out the border that the shitty windows conversion created for no reason
        im2_aligned = cv2.warpAffine(frame[5:-5,5:-5,:], warp_matrix, (S[0],S[1]), dst=im2_aligned, flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP borderMode=cv2.BORDER_TRANSPARENT)
        out.write(im2_aligned)
    cap.release()
    out.release()

if __name__ == '__main__':
    FULLNAME = sys.argv[1]
    frameStart = int(sys.argv[2])
    frameLength = int(sys.argv[3])
    path, filename = os.path.split(FULLNAME)
    noext, ext = os.path.splitext(filename)
    allTransforms=np.zeros((frameLength,3))
    outputname = noext + '_FRW' + str(frameStart) + '.avi' 
    processMovie(FULLNAME, frameStart, frameLength, outputname)
    print(outputname)
