import struct
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

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

alpha = 0.05
lamda = 0.1#alpha*alpha
input_dim = 28*28
hidden_dim1 = 300
#hidden_dim2 = 100
output_dim = 10
# initialize neural network weights
weight1 = np.random.uniform(-0.1,0.1,(input_dim+1,hidden_dim1))
weight2 = np.random.uniform(-0.1,0.1,(hidden_dim1+1,output_dim))
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

imagef = open('./data/train-images-idx3-ubyte', 'rb')
labelf = open('./data/train-labels-idx1-ubyte', 'rb')
magic, imgNum=struct.unpack(">II", imagef.read(8))
imgRow, imgCol =struct.unpack(">II", imagef.read(8))
print magic, imgNum, imgRow, imgCol
lblMagic, lblNum=struct.unpack(">II", labelf.read(8))
print lblMagic, lblNum
overallError = 0
Xa=np.zeros((imgNum, imgRow*imgCol))
Ya=np.zeros((lblNum, 1))
print '----start read data----'
for i in range(imgNum):
    Xa[i, range(imgRow*imgCol)]=np.fromfile(imagef, np.uint8, imgRow*imgCol)
    Ya[i, 0]=np.fromfile(labelf, np.uint8, 1)
#Xa=sigmoid(Xa)
#mXa=np.sum(Xa)
#Xa=Xa/mXa

for i in range(0):
    img=Xa[i].reshape(imgRow, imgCol)
    showImg(img)
    print 'y=',Ya[i]
#img=Xa[1].reshape(imgRow, imgCol)
#showImg(img)
#print 'y=',Ya[1]
#img=Xa[2].reshape(imgRow, imgCol)
#showImg(img)
#print 'y=',Ya[2]

print '----finish read data----'
valiNum=10000
loop=imgNum-valiNum
errordot=np.zeros((loop,1))

for k in range(30):
    for j in range(loop):
        X=Xa[j]
        X=X.reshape(1,imgRow*imgCol)
        #normalize
        mXa=np.max(X)
        X=X/mXa
        a1=np.c_[[1],X]
        y=Ya[j]
        #Forward propagation
        z1=np.dot(a1, weight1)
    	#print 'X=',X
	    #z1=np.asmatrix(a1)*weight1
        #a2=relu(z1)#+bias_0)
        a2=sigmoid(z1)#+bias_0)
        #print 'a2=',a2
        a21=np.c_[[1], a2]
        z3=np.dot(a21, weight2)
        a3=sigmoid(z3)#+bias_1)
        #print 'a3=',a3
        suma3=np.sum(np.abs(a3))
        #p3=softmax(f3)
        yy=np.zeros((1, output_dim))+0.01
        yy[0,y[0]]=0.99
    
        #backward propagation
        error=yy-a3
        errordot[j]=suma3
    
        gprime=sigm_deri(a3)
        #print 'delta3=',delta3.shape,'weight2.T=',weight2.T.shape,'f2=',f2.shape
        delta3=error*gprime#np.dot(delta3, weight2.T)*sigm_deri(f2)
        w2=weight2[1:,:]
        w2=w2.reshape(hidden_dim1, output_dim)
        #print  'delta3=',delta3.T,'w2.T=',w2.T.shape,'a2=',a2.shape
        #delta2=np.dot(delta3, w2.T)*relu_deri(a2)
        delta2=np.dot(delta3, w2.T)*sigm_deri(a2)
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
        weight1+=alpha*(d1)
        weight2+=alpha*(d2)
    print '----train: ',k,' finish----'
    rightSum=0
    wrongSum=0
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
        a3=sigmoid(z3)#+bias_1)
        suma3=np.sum(np.abs(a3))
        indexd=np.argmax(a3)
        if indexd==y:
            rightSum+=1
        else:
            wrongSum+=1
    
    print 'train',k,' right: ',rightSum,'Wrong: ',wrongSum
    imagef.close()
    labelf.close()
print 'overallerror=',overallError
#plt.plot(range(loop), errordot, "o")
#plt.show()
  

print 'train finish'

imagef.close()
labelf.close()
#recordf_0 = open('/home/rdeng/code/mine/nn/myparam0', 'wb')
#recordf_1 = open('/home/rdeng/code/mine/nn/myparam1', 'wb')
#recordf_2 = open('/home/rdeng/code/mine/nn/myparam2', 'wb')
#weight0.tofile(recordf_0)
#weight1.tofile(recordf_1)
#weight2.tofile(recordf_2)

testImagef = open('./data/t10k-images-idx3-ubyte', 'rb')
testLabelf = open('./data/t10k-labels-idx1-ubyte', 'rb')

tmagic, timgNum=struct.unpack(">II", testImagef.read(8))
timgRow, timgCol =struct.unpack(">II", testImagef.read(8))
print tmagic, timgNum, timgRow, timgCol
tlblMagic, tlblNum=struct.unpack(">II", testLabelf.read(8))
print tlblMagic, tlblNum
rightSum=0
wrongSum=0
for j in range(timgNum):
#for j in range(1):
    #read a 28x28 image and a byte label
    X=np.fromfile(testImagef, np.uint8, timgRow*timgCol)
    y=np.fromfile(testLabelf, np.uint8, 1)
    # where we'll store our best guess (binary encoded)
    X=Xa[j]
    X=X.reshape(1,imgRow*imgCol)
    a1=np.c_[[1],X]
    y=Ya[j]

    # where we'll store our best guess (binary encoded)
    #Forward propagation
    z1=np.dot(a1, weight1)
    a2=sigmoid(z1)#+bias_0)
    a2=np.c_[[1], a2]
    z3=np.dot(a2, weight2)
    a3=sigmoid(z3)#+bias_1)
    suma3=np.sum(np.abs(a3))
    indexd=np.argmax(a3)

    #img=X.reshape(imgRow,imgCol)
    #showImg(img)

    #print 't y=',y,'pred=',indexd

    if indexd==y:
        rightSum+=1
    else:
        wrongSum+=1

print 'right: ',rightSum,'Wrong: ',wrongSum
    

