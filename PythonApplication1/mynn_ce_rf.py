from __future__ import division
import struct
import numpy as np
import os
#import matplotlib.pyplot as plt
#import matplotlib as mpl
import math

import mynn_base as mybs
import mynn_utils as myutils
import mydefs as defs

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
        X=X.reshape(1,defs.row*defs.col)
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

do_stochastic=0
#default params
alpha = 0.003
#lamda = 0.1#alpha*alpha
row=defs.row
col=defs.col
input_dim = row*col
hidden_dim1 = 3000
hidden_dim2 = 1500
hidden_dim3 = 500
output_dim = 10
threshold=0.95
epoch=10

#layer_num=2
#layer_param1=mybs.layer_param(input_dim, hidden_dim1, sigmoid, sigm_deri, alpha)
#layer_param2=mybs.layer_param(hidden_dim1, output_dim, softmax, sigm_deri, alpha)
#layer_param=[layer_param1, layer_param2]

#layer_num=3
#layer_param1=mybs.layer_param(input_dim, hidden_dim1, relu, relu_deri, alpha)
#layer_param2=mybs.layer_param(hidden_dim1, hidden_dim2, sigmoid, relu_deri, alpha)
#layer_param3=mybs.layer_param(hidden_dim2, output_dim, softmax, sigm_deri, alpha)
#layer_param=[layer_param1, layer_param2, layer_param3]

layer_num=4
layer_param1=mybs.layer_param(input_dim, hidden_dim1, relu, relu_deri, alpha)
layer_param2=mybs.layer_param(hidden_dim1, hidden_dim2, sigmoid, relu_deri, alpha)
layer_param3=mybs.layer_param(hidden_dim2, hidden_dim3, sigmoid, sigm_deri, alpha)
layer_param4=mybs.layer_param(hidden_dim3, output_dim, softmax, sigm_deri, alpha)
layer_param=[layer_param1, layer_param2, layer_param3, layer_param4]

nnetwork = mybs.nnetwork(layer_num, layer_param, input_dim, output_dim, epoch, threshold)

if (do_stochastic==0):
    print('----try before train----')
    #timgNum, timgRow, timgCol, Xt = myutils.load_image_data('t10k-images.idx3-ubyte', 0)
    #tlblNum, Yt = myutils.load_label_data('t10k-labels.idx1-ubyte',0)
    timgRow=row
    timgCol=col
    timgNum, Xt = myutils.load_real_image_data('img_data_validate', row, col)
    tlblNum, Yt = myutils.load_real_label_data('img_label_validate', timgNum)
    #Xt=255-Xt
    Xt=mybs.normalize(Xt)
    right_num, wrong_num = nnetwork.predict_validate(Xt, Yt, timgNum)
    test_accuracy=right_num/(right_num+wrong_num)
    print ('----test accuracy=', test_accuracy)
    if (test_accuracy>threshold):
        print('no need to train')
    else:
        print('----start reading training data----')
        #this is for MNIST data base
        #imgNum, imgRow, imgCol, Xa = myutils.load_image_data('train-images.idx3-ubyte', 0)
        #lblNum, Ya = myutils.load_label_data('train-labels.idx1-ubyte',0)
        #this is for camera captured
        imgRow=row
        imgCol=col
        imgNum, Xa = myutils.load_real_image_data('img_data', row, col)
        lblNum, Ya = myutils.load_real_label_data('lbl_data',imgNum)
        Xa=mybs.normalize(Xa)
        valiNum=int(imgNum/10)
        Xv=Xa[imgNum-valiNum-1:imgNum,:]
        Yv=Ya[lblNum-valiNum-1:lblNum,:]
        print ('----finish read training data----')

        nnetwork.train(Xa, Ya, imgNum, Xt, Yt, timgNum)

        print ('----nnetwork.train finish----')
        print ('----start test----')

        right_num, wrong_num = nnetwork.predict_validate(Xt, Yt, timgNum)
        test_accuracy=right_num/(right_num+wrong_num)

        print ('----test accuracy=', test_accuracy)

        nnetwork.dump2file()
else:
    print('=======train incrementaly=====')
    nnetwork.reinit_weight_rand()
    

