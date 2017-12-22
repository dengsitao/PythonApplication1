from __future__ import division
import struct
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

import mynn_base as mybs
import mynn_utils as myutils
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

def blank_deri(x):
    return x

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

def forwardProp(X, weight, activate_func):
    X1=np.c_[1, X]
    Z1=np.dot(X1, weight)
    Res1 = activate_func(Z1)
    return Res1

# def normalize(X):
#     return (X-np.mean(X))/np.std(X)

def predict(Xi, Yi, weight1, weight2):
    num=Yi.size
    rightSum=0
    wrongSum=0
    for j in range(num):
        #for j in range(1):
        #read a 28x28 image and a byte label
        #X=Xa[j+imgNum-valiNum]
        X=Xi[j]
        X=X.reshape(1,28*28)
        y=Yi[j]
        #y=Ya[j+imgNum-valiNum]
        #Forward propagation
        a2=forwardProp(X, weight1, sigmoid)
        #a2=forwardProp(X, weight1, relu)
        a3=forwardProp(a2, weight2, softmax)
        indexd=np.argmax(a3)
        if indexd==y:
            rightSum+=1
        else:
            wrongSum+=1

    accuracy=rightSum/num
    print ('predict',' right: ',rightSum,'Wrong: ',wrongSum, accuracy*100, '%')
    return accuracy

alpha = 0.003
lamda = 0.1#alpha*alpha
input_dim = 28*28
hidden_dim1 = 300
#hidden_dim2 = 100
output_dim = 10
# initialize neural network weights
weight1 = np.random.uniform(-0.1,0.1,(input_dim+1,hidden_dim1))
weight2 = np.random.uniform(-0.1,0.1,(hidden_dim1+1,output_dim))

bias_1=np.zeros((1, hidden_dim1))
bias_2=np.zeros((1, output_dim))

d1=np.zeros(weight1.shape)
d2=np.zeros(weight2.shape)

imgNum, imgRow, imgCol, Xa = myutils.load_image_data('train-images.idx3-ubyte', 0)
lblNum, Ya = myutils.load_label_data('train-labels.idx1-ubyte',0)
Xa=mybs.normalize(Xa)
# print 'X1', imgNum1, imgRow1, imgCol1, Xa1.shape, 'Y1', lblNum1, Ya1.shape
# imgNum, imgRow, imgCol, lblNum, Xa, Ya = myutils.loadMNISTData()
# print 'X', imgNum, imgRow, imgCol, Xa.shape, 'Y', lblNum, Ya.shape
# for i in range(10):
#     img=Xa[i].reshape(imgRow, imgCol)
#     showImg(img)
#     print 'Ya[',i,']=', Ya[i]
# for i in range(imgNum):
#     for j in range(imgRow*imgCol):
#         if Xa1[i,j] == Xa[i,j]:
#             continue
#         else:
#             print 'Xa[',i,',',j,'] !=', 'Xa1[',i,',',j,']'
#             print 'Xa=', Xa[i,:]
#             print 'Xa1', Xa1[i,:]
#             raw_input('wait')

#     if Ya1[i] != Ya[i]:
#         print 'Ya[',i,'] !=', 'Ya1[',i,']'
#         print 'Ya=', Ya[i]
#         print 'Ya1', Ya1[i]

epoch=10
valiNum=int(imgNum/10)
Xv=Xa[imgNum-valiNum-1:imgNum,:]
Yv=Ya[lblNum-valiNum-1:lblNum,:]

# layer1=mybs.layer(input_dim, hidden_dim1, sigmoid, sigm_deri, alpha)
# layer2=mybs.layer(hidden_dim1, output_dim, softmax, sigm_deri, alpha)



layer_param1=mybs.layer_param(input_dim, hidden_dim1, sigmoid, sigm_deri, alpha)
layer_param2=mybs.layer_param(hidden_dim1, output_dim, softmax, sigm_deri, alpha)
# layer_param1=mybs.layer_param(input_dim, hidden_dim1, relu, relu_deri, alpha)
# layer_param2=mybs.layer_param(hidden_dim1, output_dim, softmax, relu_deri, alpha)
layer_param=[layer_param1, layer_param2]
nnetwork = mybs.nnetwork(Xa, Ya, imgNum, 2, layer_param, input_dim, output_dim, epoch)

print ('----finish read data----')


nnetwork.train()

print ('----nnetwork.train finish----')

# loop=imgNum-valiNum
# errordot=np.zeros((loop*epoch,1))

# for k in range(epoch):
#     overallError=0
#     for j in range(loop):
#         X=Xa[j]
#         X=X.reshape(1,imgRow*imgCol)
#         #print 'Xa.shape=',Xa[j].shape,'X.shape=',X.shape
#         y=Ya[j]
#         #Forward propagation
#         #a2=forwardProp(X, weight1, sigmoid)
#         a2=layer1.forward_prop(X)

#         #p3=forwardProp(a2, weight2, softmax)
#         p3=layer2.forward_prop(a2)

#         yy=np.zeros((1, output_dim))
#         yy[0.0,y[0]]=1.0
    
#         #backward propagation
#         error=p3-yy
#         errordot[k*loop+j]=-math.log(p3[0, y[0]])
#         overallError+=errordot[k*loop+j]
        
#         delta1=layer2.backward_prop(error)
#         delta=layer1.backward_prop(delta1)

#     print '----train: ',k,' finish----'
#     print 'overallerror=',overallError
#     #accuracy=predict(Xv, Yv, weight1, weight2)
#     accuracy=predict(Xv, Yv, layer1.lyr_weight, layer2.lyr_weight)

#     if (accuracy > 0.95):
#         break
#     #plt.plot(range(loop), errordot, "o")
#     #plt.show()
# #print 'overallerror=',overallError
# #plt.plot(range(epoch*loop), errordot)
# #plt.show()
  

print ('train finish')

#imagef.close()
#labelf.close()
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
#Xt=(Xt-np.mean(Xt))/np.std(Xt)
Xt=mybs.normalize(Xt)
# test_accuracy=predict(Xt, Yt, nnetwork.layers[0].lyr_weight, nnetwork.layers[1].lyr_weight)
right_num, wrong_num = nnetwork.predict(Xt, Yt, timgNum)
test_accuracy=right_num/(right_num+wrong_num)

print ('test accuracy=', test_accuracy)
    

