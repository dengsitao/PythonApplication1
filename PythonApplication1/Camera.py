import numpy as np
import cv2

#----my nn include---
import mynn_ce_rf

cv2.namedWindow("preview")
vc = cv2.VideoCapture(1)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")