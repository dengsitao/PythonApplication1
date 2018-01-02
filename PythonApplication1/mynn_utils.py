import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib as mpl

import mynn_base as mybs
import mydefs as defs

#row=defs.row
#col=defs.col*2


def load_real_image_data(file_name, row, col):
    img_file=open(file_name, 'rb')
    data=np.fromfile(img_file)
    img_num=int(data.shape[0]/row/col)
    data=data.reshape(img_num, row*col)
    img_file.close()
    return img_num, data
    
def load_real_label_data(file_name, count):
    lblfile=open(file_name, 'rb')
    Ya=np.zeros((count, 1))
    for i in range(count):
        #Ya[i, 0]=np.fromfile(file_name, np.uint8, 1)
        ytemp=lblfile.read(1)
        #print('count=',str(count),'  i=',str(i))
        #if (len(ytemp)==0):
        #    print('a zero length ',str(i))
        #else:
        Ya[i, 0]=ord(ytemp)
        #print('read ',str(i),' =',Ya[i,0])
    lblfile.close()
    return count, Ya

def load_image_data(file_name, offset):
    #imgfile = open('./data/train-images-idx3-ubyte', 'rb')
    img_file = open(file_name, 'rb')
    magic, imgNum, imgRow, imgCol = struct.unpack(">IIII", img_file.read(16))
    print ('Loading Image File: magic=', magic, 'imgNum=', imgNum, 'imgRow=', imgRow, 'imgCol=', imgCol,'offset=',offset)
    count=imgNum-offset
    Xa=np.zeros((count, imgRow*imgCol))
    #img_file.seek(offset*imgRow*imgCol+16)
    for i in range(count):
        Xa[i, range(imgRow*imgCol)]=np.fromfile(img_file, np.uint8, imgRow*imgCol)
    img_file.close()
    return imgNum, imgRow, imgCol, Xa

def load_label_data(file_name, offset):
    #lblfile = open('./data/train-labels-idx1-ubyte', 'rb')
    lblfile = open(file_name, 'rb')
    magic, lblNum = struct.unpack(">II", lblfile.read(8))
    print ('Loading Label File: magic=', magic, 'lblNum=', lblNum, 'offset=',offset)
    count=lblNum-offset
    Ya=np.zeros((count, 1))
    #lblfile.seek(offset+8)
    for i in range(count):
        Ya[i, 0]=np.fromfile(lblfile, np.uint8, 1)
    lblfile.close()
    return lblNum, Ya

def loadImgData(imgfile, imgNum, count, offset, imgRow, imgCol):
    if count > imgNum:
        print ('count=',count,' > ','imgNum=', imgNum)
        count=imgNum
    Xa=np.zeros((count, imgRow*imgCol))
    imgfile.seek(offset*imgRow*imgCol+16)
    for i in range(count):
        Xa[i, range(imgRow*imgCol)]=np.fromfile(imgfile, np.uint8, imgRow*imgCol)
    return Xa

def loadLabelData(lblfile, imgNum, count, offset, imgRow, imgCol):
    if count > imgNum:
        print ('count=',count,' > ','lblNum=', imgNum)
        count=imgNum
    Ya=np.zeros((count, 1))
    lblfile.seek(offset+8)
    for i in range(count):
        Ya[i, 0]=np.fromfile(lblfile, np.uint8, 1)
    return Ya

def loadMNISTData():
    imgfile = open('train-images-idx3-ubyte', 'rb')
    magic, imgNum, imgRow, imgCol = struct.unpack(">IIII", imgfile.read(16))
    print ('Image File: magic=', magic, 'imgNum=', imgNum, 'imgRow=', imgRow, 'imgCol=', imgCol)
    trainNum=imgNum
    Xa=loadImgData(imgfile, imgNum, trainNum, 0, imgRow, imgCol)


    lblfile = open('train-labels-idx1-ubyte', 'rb')
    magic, lblNum = struct.unpack(">II", lblfile.read(8))
    print ('Label File: magic=', magic, 'lblNum=', lblNum)
    Ya=loadLabelData(lblfile, lblNum, trainNum, 0, imgRow, imgCol)

    imgfile.close()
    lblfile.close()
    Xa=mybs.normalize(Xa)
    #for i in range(10):
    #img=Xa[i].reshape(imgRow, imgCol)
    #showImg(img)
    #print 'Ya[',i,']=', Ya[i]
    return imgNum, imgRow, imgCol, lblNum, Xa, Ya
    
def showImg(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    #from matplotlib import pyplot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()