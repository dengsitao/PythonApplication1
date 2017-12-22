from __future__ import division
import struct
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import time,datetime

np.random.seed(0)
## compute sigmoid nonlinearity
def sigmoid(x):
    output = 1.0/(1.0+np.exp(-x))
    return output
# convert output of sigmoid function to its derivative
def sigm_deri(output):
    return output*(1-output)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    #exp_scores=np.exp(x)
    #probs=exp_scores/np.sum(exp_scores, axis=1,keepdims=True)
    #return probs

def softmax_deri(signal):
    J = - signal[..., None] * signal[:, None, :] # off-diagonal Jacobian
    iy, ix = np.diag_indices_from(J[0])
    J[:, iy, ix] = signal * (1. - signal) # diagonal
    return J.sum(axis=1) # sum across-rows for each sample

def relu(x):
    return np.maximum(x, 0)

def relu_deri(output):
    return 1.*(output>0)


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

alpha = 0.003
lamda = 0.1#alpha*alpha
input_dim = 28*28
hidden_dim1 = 300
#hidden_dim2 = 100
output_dim = 10
# initialize neural network weights
weight1 = np.random.uniform(-0.1,0.1,(input_dim+1,hidden_dim1))
weight2 = np.random.uniform(-0.1,0.1,(hidden_dim1+1,output_dim))
#imgw=weight1.reshape(1, (input_dim+1)*hidden_dim1)
#showImg(imgw)
#for i in range(785):
#    weight1[i,:]=-0.1+i*0.0003
#for i in range(301):
#    weight2[i,:]=-0.1+i*0.004

#print 'weight1=',weight1
#print 'weight2=',weight2
#weight2 = (2*np.random.random((hidden_dim2+1,output_dim)) -1)
bias_1=np.zeros((1, hidden_dim1))
bias_2=np.zeros((1, output_dim))
#bias_2=np.zeros((1, output_dim))
d1=np.zeros(weight1.shape)
d2=np.zeros(weight2.shape)
#d2=np.zeros(weight2.shape)

imagef = open('train-images.idx3-ubyte', 'rb')
labelf = open('train-labels.idx1-ubyte', 'rb')
magic, imgNum=struct.unpack(">II", imagef.read(8))
imgRow, imgCol =struct.unpack(">II", imagef.read(8))
print(magic, imgNum, imgRow, imgCol)
lblMagic, lblNum=struct.unpack(">II", labelf.read(8))
print (lblMagic, lblNum)
overallError = 0
Xa=np.zeros((imgNum, imgRow*imgCol))
Ya=np.zeros((lblNum, 1))
print('----start read data----')
for i in range(imgNum):
    Xa[i, range(imgRow*imgCol)]=np.fromfile(imagef, np.uint8, imgRow*imgCol)
    Ya[i, 0]=np.fromfile(labelf, np.uint8, 1)
#Xa=sigmoid(Xa)
#Xa=Xa/255
Xa=(Xa-np.mean(Xa))/np.std(Xa)

for i in range(0):
    img=Xa[i].reshape(imgRow, imgCol)
    showImg(img)
    print ('y=',Ya[i])
#img=Xa[1].reshape(imgRow, imgCol)
#showImg(img)
#print 'y=',Ya[1]
#img=Xa[2].reshape(imgRow, imgCol)
#showImg(img)
#print 'y=',Ya[2]
imagef.close()
labelf.close()

print ('----finish read data----')
valiNum=10000
loop=imgNum-valiNum
errordot=np.zeros((loop,1))

for k in range(3):
    overallError=0
    time1=time.time()
    for j in range(loop):
        X=Xa[j]
        X=X.reshape(1,imgRow*imgCol)
        a1=np.c_[[1],X]
        y=Ya[j]
        #Forward propagation
        z1=np.dot(a1, weight1)
	#print 'X=',X
	#z1=np.asmatrix(a1)*weight1
        a2=sigmoid(z1)#+bias_0)
        #print 'a2=',a2
        a21=np.c_[[1], a2]
        z3=np.dot(a21, weight2)
        a3=sigmoid(z3)#+bias_1)
        #print 'a3=',a3
        suma3=np.sum(np.abs(a3))
	#print 'a3=',a3
	#print 'suma3=',suma3
        p3=softmax(z3)
        #yy=np.zeros((1, output_dim))+0.1/9
        #yy[0,y[0]]=0.9
        yy=np.zeros((1, output_dim))
        #print(y[0])
        yy[0,int(y[0])]=1.0
    
        #backward propagation
	#error=pow(a3-yy,2)/2
        #error=yy-p3
        #error=a3-yy
        error=p3-yy
        errordot[j]=-math.log(p3[0, int(y[0])])
        overallError+=errordot[j]
	#gprime=softmax_deri(p3)
	#delta4=error*gprime
    
        gprime=sigm_deri(a3)
        #print 'delta3=',delta3.shape,'weight2.T=',weight2.T.shape,'f2=',f2.shape
        delta3=error#*gprime
        w2=weight2[1:,:]
        w2=w2.reshape(hidden_dim1, output_dim)
        #print  'delta3=',delta3.T,'w2.T=',w2.T.shape,'a2=',a2.shape
        #print 'back prop'
        delta2=np.dot(delta3, w2.T)*sigm_deri(a2)

        # w1=weight1[1:,:]
        # w1=w1.reshape(input_dim, hidden_dim1)
        # delta1=np.dot(delta2, w1.T)*sigm_deri(X)

        #print 'delta2=',delta2.shape,'weight2.T=',weight2.T.shape,'a3=',a3.shape
        #delta1=np.dot(delta2, weight1.T)*sigm_deri(a2)
        #print 'd2=',d2.shape,'delta3.T=',delta3.T.shape,'f2=',f2.shape
        #print 'weight2=',weight2.shape,'d2=',d2.shape,'delta3.T=',delta3.T.shape,'a2=', a2.shape
        d2=np.dot(delta3.T, a21).T
        #dbias_1=delta2
        #print 'weight1=',weight1.shape,'d1=',d1.shape,'delta2.T=',delta2.T.shape,'a1=', a1.shape
        #delta2=delta2[:,1:]
        #delta2=delta2.reshape(1, hidden_dim1)
        d1=np.dot(delta2.T, a1).T
        #weight1-=alpha*(d1)#+lamda*weight1)
        #weight2-=alpha*(d2)#+lamda*weight2)
        weight1-=alpha*(d1)#+lamda*weight1)
        weight2-=alpha*(d2)#+lamda*weight2)
    print ('----train: ',k,' finish----')
    print ('training',k,' use ',time.time()-time1)
    print ('overallerror=',overallError)
    rightSum=0
    wrongSum=0
    time1=time.time()
    for j in range(valiNum):
    #for j in range(1):
        #read a 28x28 image and a byte label
        X=Xa[j+imgNum-valiNum]
        X=X.reshape(1,imgRow*imgCol)
        a1=np.c_[[1],X]
        y=Ya[j+imgNum-valiNum]

        #img=X.reshape(imgRow, imgCol)
        #showImg(img)
        #print 'v y=',y
        # where we'll store our best guess (binary encoded)
        #Forward propagation
        z1=np.dot(a1, weight1)
        a2=sigmoid(z1)#+bias_0)
        a2=np.c_[[1], a2]
        z3=np.dot(a2, weight2)
        #a3=sigmoid(z3)#+bias_1)
        #suma3=np.sum(np.abs(a3))
        a3=softmax(z3)
        indexd=np.argmax(a3)
        if indexd==y:
            rightSum+=1
        else:
            wrongSum+=1
    print ('validating',k,' use ',time.time()-time1)
    print ('train',k,' right: ',rightSum,'Wrong: ',wrongSum, 'accuracy=',rightSum/valiNum, '%')
    if (rightSum/valiNum > 0.95):
        break
    #plt.plot(range(loop), errordot, "o")
    #plt.show()
#print 'overallerror=',overallError
#plt.plot(range(loop), errordot, "o")
#plt.show()
  

print ('train finish')

imagef.close()
labelf.close()
#recordf_0 = open('/home/rdeng/code/mine/nn/myparam0', 'wb')
#recordf_1 = open('/home/rdeng/code/mine/nn/myparam1', 'wb')
#recordf_2 = open('/home/rdeng/code/mine/nn/myparam2', 'wb')
#weight0.tofile(recordf_0)
#weight1.tofile(recordf_1)
#weight2.tofile(recordf_2)

timagef = open('t10k-images.idx3-ubyte', 'rb')
tlabelf = open('t10k-labels.idx1-ubyte', 'rb')

tmagic, timgNum=struct.unpack(">II", timagef.read(8))
timgRow, timgCol =struct.unpack(">II", timagef.read(8))
print (tmagic, timgNum, timgRow, timgCol)
tlblMagic, tlblNum=struct.unpack(">II", tlabelf.read(8))
print (tlblMagic, tlblNum)

Xt=np.zeros((timgNum, timgRow*imgCol))
Yt=np.zeros((tlblNum, 1))
print ('----start read data----')
for i in range(timgNum):
    Xt[i, range(timgRow*timgCol)]=np.fromfile(timagef, np.uint8, timgRow*timgCol)
    Yt[i, 0]=np.fromfile(tlabelf, np.uint8, 1)
#Xt=sigmoid(Xt)
Xt=(Xt-np.mean(Xt))/np.std(Xt)
rightSum=0
wrongSum=0
for j in range(timgNum):
#for j in range(1):
    #read a 28x28 image and a byte label
    # where we'll store our best guess (binary encoded)
    X=Xt[j]
    X=X.reshape(1,timgRow*timgCol)
    a1=np.c_[[1],X]
    y=Yt[j]

    # where we'll store our best guess (binary encoded)
    #Forward propagation
    z1=np.dot(a1, weight1)
    a2=sigmoid(z1)#+bias_0)
    a2=np.c_[[1], a2]
    z3=np.dot(a2, weight2)
    #a3=sigmoid(z3)#+bias_1)
    a3=softmax(z3)#+bias_1)
    #suma3=np.sum(np.abs(a3))
    indexd=np.argmax(a3)

    #img=X.reshape(imgRow,imgCol)
    #showImg(img)

    #print 't y=',y,'pred=',indexd

    if indexd==y:
        rightSum+=1
    else:
        wrongSum+=1

print('right: ',rightSum,'Wrong: ',wrongSum)
    

