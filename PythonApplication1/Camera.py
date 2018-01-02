#import numpy as np
import cv2
import enzyme
import fourcc
#----my nn include---
import mynn_ce_rf as nn
import numpy as np
import mydefs as defs

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
x=defs.start_x
y=defs.start_y
row=defs.row
col=defs.col
#datafile=open('img_data_validate', 'ab')
#labelfile=open('img_label_validate', 'ab')
datafile=open('img_data', 'ab')
labelfile=open('lbl_data', 'ab')
crop_image1=np.zeros((row,col))
crop_image2=np.zeros((row,col))
crop_image3=np.zeros((row,col))
crop_image4=np.zeros((row,col))
while rval:
    #cv2.imshow("preview", frame)
    rval, frame = vc.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame[x:x+row,y:y+1]=255
    frame[x:x+row,y+col:y+1+col]=255
    frame[x:x+row,y+col*2:y+1+col*2]=255
    frame[x:x+row,y+col*3:y+1+col*3]=255
    frame[x:x+row,y+col*4:y+col*4+1]=255
    frame[x:x+1,y:y+col*4]=255
    frame[x+row:x+row+1,y:y+col*4]=255
    gray=255-gray
    rec_image=gray[x:x+row,y:y+col*4]
    #crop_image1[0:row,7:21]=rec_image[0:row,7:7+14]
    #crop_image2[0:row,7:21]=rec_image[0:row,7+14:7+14+14]
    #crop_image3[0:row,7:21]=rec_image[0:row,7+14+14:7+14+14+14]
    #crop_image4[0:row,7:21]=rec_image[0:row,7+14+14+14:7+14+14+14+14]
    crop_image1[0:row,0:col]=rec_image[0:row,0:col]
    crop_image2[0:row,0:col]=rec_image[0:row,col:col*2]
    crop_image3[0:row,0:col]=rec_image[0:row,col*2:col*3]
    crop_image4[0:row,0:col]=rec_image[0:row,col*3:col*4]
    
    pred1=nn.nnetwork.predict(crop_image1)#.reshape(1, 28*28))
    pred2=nn.nnetwork.predict(crop_image2)
    pred3=nn.nnetwork.predict(crop_image3)
    pred4=nn.nnetwork.predict(crop_image4)
    print('predict: ',str(pred1),' ',str(pred2),' ',str(pred3),' ',str(pred4))
    cv2.imshow("preview", frame)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    if key == ord('p'): # p to pause and show a image
        nn.myutils.showImg(crop_image1)
        nn.myutils.showImg(crop_image2)
        nn.myutils.showImg(crop_image3)
        nn.myutils.showImg(crop_image4)
    if key == ord('i'):
        act_y=input("input number: ")
        y1=int(act_y[0])
        y2=int(act_y[1])
        y3=int(act_y[2])
        y4=int(act_y[3])
        #print('y1='+str(y1)+' y2='+str(y2)+' y3='+str(y3)+' y4='+str(y4))
        pred_y1=nn.nnetwork.train_stochastic(crop_image1, y1)
        pred_y2=nn.nnetwork.train_stochastic(crop_image2, y2)
        pred_y3=nn.nnetwork.train_stochastic(crop_image3, y3)
        pred_y4=nn.nnetwork.train_stochastic(crop_image4, y4)

        print('correct: ',str(pred_y1),' ',str(pred_y2),' ',str(pred_y3),' ',str(pred_y4))
        crop_image1.tofile(datafile)
        crop_image2.tofile(datafile)
        crop_image3.tofile(datafile)
        crop_image4.tofile(datafile)
        labelfile.write(y1.to_bytes(1, 'big'))
        labelfile.write(y2.to_bytes(1, 'big'))
        labelfile.write(y3.to_bytes(1, 'big'))
        labelfile.write(y4.to_bytes(1, 'big'))

datafile.close()
labelfile.close()
cv2.destroyWindow("preview")