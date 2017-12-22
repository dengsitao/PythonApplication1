import numpy as np
import cv2

#----my nn include---
import mynn_ce_rf as nn

cv2.namedWindow("preview")
vc = cv2.VideoCapture(1)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
x=30
y=30
while rval:
    #cv2.imshow("preview", frame)
    rval, frame = vc.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #crop_image=frame[0:0+28,0,0+28];
    frame[x:x+28,y:y+1]=255
    frame[x:x+28,y+28:y+28+1]=255
    frame[x:x+1,y:y+28]=255
    frame[x+28:x+28+1,y:y+28]=255
    crop_image=gray[x:x+28,y:y+28]
    pred=nn.nnetwork.predict(crop_image)#.reshape(1, 28*28))
    print('pred='+str(pred))
    cv2.imshow("preview", gray)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")