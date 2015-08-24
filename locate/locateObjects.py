
import cv2
import cv
import numpy as np
import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('../fhgClassifier/')

from SimpleCV import SVMClassifier
from SimpleCV import Image, Display
from circularHOGExtractor import circularHOGExtractor


cap = cv2.VideoCapture('/home/ctorney/data/wildebeest/from_grant/GOPR0076.MP4')
cap.set(cv.CV_CAP_PROP_POS_FRAMES,10000)

ex = int(cap.get(cv.CV_CAP_PROP_FOURCC))
#     outputVideo.open(outMovie, ex, cap.get(CV_CAP_PROP_FPS), S, true);
S = (int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)))

   
out = cv2.VideoWriter('/home/ctorney/Dropbox/output.avi', cv.CV_FOURCC('M','J','P','G'), cap.get(cv.CV_CAP_PROP_FPS), S, True)

params = cv2.SimpleBlobDetector_Params()
params.minDistBetweenBlobs = 1.0;
params.filterByInertia = False;
params.filterByConvexity = False;
params.filterByColor = False;
params.filterByCircularity = False;
params.filterByArea = True;
params.minArea = 5.0;
params.maxArea = 100.0;
params.minThreshold = 15;
params.maxThreshold = 255;

b = cv2.SimpleBlobDetector(params)

cl = SVMClassifier.load('../trainClassifier/svmWildebeest.xml')
classes = []
classes.append('yes')
classes.append('no')

box_dim = 48

ret, oldframe = cap.read()
ret, frame = cap.read()
M_ALL = cv2.estimateRigidTransform(oldframe,frame,0)
M_ALL = np.vstack([M_ALL,(0,0,1)])
print M_ALL
for tt in range(1500):
    print tt
# Capture frame-by-frame
    ret, frame = cap.read()

    blob = b.detect(frame)
    M = cv2.estimateRigidTransform(frame,oldframe,0)
    M_ALL = np.dot(np.vstack([M,(0,0,1)]),M_ALL)
    np.copyto(oldframe,frame)

    for beest in blob:
        tmpImg = Image(frame, cv2image=True).crop(int(beest.pt[0]),int(beest.pt[1]),box_dim,box_dim,centered=True)
        if ((tmpImg.width + tmpImg.height) == 2*box_dim):
            isit = cl.classify(tmpImg)
            if isit=='yes':
                cv2.circle(frame, ((int(beest.pt[0]), int(beest.pt[1]))),4,255,1)
#cv2.drawKeyPoints(frame, beest, frame)

        # Our operations on the frame come here
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
    showframe = cv2.warpAffine(frame,M_ALL[0:2,:],S) 
 #/   cv2.imshow('frame',showframe)
    out.write(showframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

