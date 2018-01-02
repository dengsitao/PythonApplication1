import numpy as np
import mynn_utils as myutils
import math

import mydefs as defs

#used for manually tag data
def tag_data(data_file, label_file, col, row):
    data=open(data_file, 'rb')
    label=open(label_file, 'ab')
    Xa=np.fromfile(data)
    img_num=Xa.shape[0]/col/row
    Xa=Xa.reshape(int(img_num), col*row)
    for i in range(int(img_num)):
        img1=Xa[i]
        img1=img1.reshape((col, row))
        print(img1.shape)
        myutils.showImg(img1)
        value=input("number: ")
        int_value=int(value)
        label.write(int_value.to_bytes(1, 'big'))
    data.close()
    label.close()

#dup samples by 
# 1. move uppper line of pixel to bottom
# 2. tilt( how?)
def dup_data(data_file, label_file, col, row, flag2file):
    data=open(data_file, 'rb')
    out_data=open(data_file+'_out', 'ab')
    label=open(label_file, 'rb')
    out_lbl=open(label_file+'_out', 'ab')
    Xa=np.fromfile(data)
    Xa=255-Xa
    img_num=int(Xa.shape[0]/col/row)
    Xa=Xa.reshape(img_num, col*row)
    img=np.zeros((col, row))
    img_1=np.zeros((col, row))
    ver_step=3
    hor_step=3
    theta=5;#degree
    luma_step=5
    margin=int(col/4)
    act_col=int(col/2)
    for i in range(img_num):
        img=Xa[i]
        Ya=np.fromfile(label, np.uint8, 1)
        int_value=int(Ya)
        img=img.reshape(col, row)
        img_1=np.copy(img)
        
        if (flag2file==1):
            img.tofile(out_data)
            out_lbl.write(int_value.to_bytes(1, 'big'))
        #myutils.showImg(img)
        for l in range(luma_step*2):
            img_1=img_1-l*3+luma_step
            img_1[img_1<0]=0
            if (flag2file==1):
                img.tofile(out_data)
                out_lbl.write(int_value.to_bytes(1, 'big'))
            #myutils.showImg(img)
            for j in range(ver_step):
                img=np.copy(img_1)
                img=img.reshape(col, row)
                row1=img[0:j,0:col]
                img[0:row-j,0:col]=img[j:row,0:col] #move the image up j pixel
                img[row-j:row,0:col]=row1

                #myutils.showImg(img)
                if (flag2file==1):
                    img.tofile(out_data)
                    out_lbl.write(int_value.to_bytes(1, 'big'))
        
            for k in range(hor_step):
                img=np.copy(img_1)
                img=img.reshape(col, row)
                #myutils.showImg(img)
                col1=img[0:row,margin:margin+k]
                img[0:row,margin:margin+act_col-k]=img[0:row,margin+k:margin+act_col] #move the image right k pixel
                img[0:row,margin+act_col-k:margin+act_col]=col1
                #myutils.showImg(img)
                if (flag2file==1):
                    img.tofile(out_data)
                    out_lbl.write(int_value.to_bytes(1, 'big'))

            img=np.copy(img_1)
            img=img.reshape(col, row)
            #myutils.showImg(img)  
            for h in range(row):
                step=(h)*math.tan(theta*math.pi/180)
                #print('step=',str(step),'int step=',str(int(step)))
                istep=int(step)
                col1=img[h:h+1,margin:margin+istep]
                img[h:h+1,margin:margin+act_col-istep]=img[h:h+1,margin+istep:margin+act_col] #tilt the image by theta degree
                img[h:h+1,margin+act_col-istep:margin+act_col]=col1
            if (flag2file==1):
                img.tofile(out_data)
                out_lbl.write(int_value.to_bytes(1, 'big'))
            #myutils.showImg(img)

    data.close()
    label.close()
    out_data.close()
    out_lbl.close()

#dup_data('img_data', 'lbl_data', 28, 28, 1)
#data=np.zeros((3,3))

def read_and_show():
    imgNum, Xa = myutils.load_real_image_data('img_data_validate')
    lblNum, Ya = myutils.load_real_label_data('img_label_validate',imgNum)
    print(Xa.shape)
    for i in range(5):
        img=Xa[i]
        img=img.reshape(row,col)
        print('y=',str(Ya[i]))
        myutils.showImg(img)

#frame=np.ones((128,128))
#print(frame.shape)
#datafile=open('img_data','ab')
#crop_image1=np.zeros((row,col))
#crop_image2=np.zeros((row,col))
#crop_image3=np.zeros((row,col))
#crop_image4=np.zeros((row,col))
#crop_image1=frame[0:56,0:28]
#crop_image2=frame[0:56,0:28]
#crop_image3=frame[0:56,0:28]
#crop_image4=frame[0:56,0:28]
#crop_image1.tofile(datafile)
#crop_image2.tofile(datafile)
#crop_image3.tofile(datafile)
#crop_image4.tofile(datafile)
#datafile.close()







def trim_data():
    imgNum, Xa = myutils.load_real_image_data('img_data_validate', defs.row, defs.col*2)
    print('imgNm=',str(imgNum))
    real_data=np.zeros((imgNum,defs.row*defs.col))
    for i in range(imgNum):
        tdata=Xa[i].reshape(defs.row, defs.col*2)
        rdata=real_data[i].reshape(defs.row, defs.col)
        rdata=np.copy(tdata[:, 7:21])
        real_data[i]=rdata.reshape(1, defs.row*defs.col)
    #print(real_data.shape)

    wtfile=open('img_data_validate_trim', 'ab')
    real_data.tofile(wtfile)
    wtfile.close()

    rtfile=open('img_data_validate_trim', 'rb')
    data=np.fromfile(rtfile)
    data=data.reshape(imgNum, defs.row*defs.col)
    for i in range(3):
        myutils.showImg(data[i].reshape(defs.row, defs.col))
    rtfile.close()

imgNum, Xa = myutils.load_real_image_data('img_data', defs.row, defs.col)

print('imgNm=',str(imgNum))
lblfile=open('lbl_data', 'rb')
y=np.zeros((imgNum))
for i in range(imgNum):
    y[i]=ord(lblfile.read(1))
lblfile.close()

for i in range(10):
    index=i*10+1
    #y=ord(lblfile.read(1))
    print('y=',str(y[index]))
    myutils.showImg(Xa[index].reshape(defs.row,defs.col))


#for i in range(imgNum):
#    #Ya[i, 0]=np.fromfile(file_name, np.uint8, 1)
#    Ya[i, 0]=ord(lblfile.read(1))
#    print('read ',str(i),' =',Ya[i,0])
#lblfile.close()
#myutils.showImg(Xa[0].reshape(row,col))
#myutils.showImg(Xa[3].reshape(row,col))
#myutils.showImg(Xa[imgNum-2].reshape(row,col))
#myutils.showImg(Xa[imgNum-1].reshape(row,col))


