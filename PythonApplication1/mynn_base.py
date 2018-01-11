from __future__ import division
import numpy as np
import time,datetime
import os.path
import mynn_utils as util
import mydefs as defs

def normalize(X):
    return (X-np.mean(X))/np.std(X)

class layer:
    'one layer with neureons(X), weights and activate functions'
    input_dim = 0
    output_dim = 0
    alpha = 0.01

    def __init__(self, input_dim, output_dim, act_func, act_func_deri, alpha, id):
        self.alpha=alpha
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.activate_function=act_func
        self.d_act_func=act_func_deri
        self.id=id
        if(os.path.isfile('weight_lyr_'+str(id))):
            self.lyr_weight=np.fromfile('weight_lyr_'+str(id))
            self.lyr_weight=self.lyr_weight.reshape(input_dim+1,output_dim)
            print('init layer: '+str(id) + ' with existing weights '+str(input_dim)+'x'+str(output_dim))
        else:
            self.lyr_weight=np.random.uniform(-0.1,0.1,(input_dim+1,output_dim))
            print('init layer: '+str(id) + ' with random numbers')

    def reinit_weight_rand(self):
        self.lyr_weight=np.random.uniform(-0.1,0.1,(self.input_dim+1,self.output_dim))
        print('init layer: '+str(self.id) + ' with random numbers')

    def forward_prop(self, X):
        self.X=X
        X1=np.c_[1, X]
        a=np.dot(X1, self.lyr_weight)
        self.a1=self.activate_function(a)
        return self.a1

    def calc_delta(self, delta):
        w2=self.lyr_weight[1:,:]
        w2=w2.reshape(self.input_dim, self.output_dim)
        #print  'input_dim=',self.input_dim,'output_dim=',self.output_dim,'delta.shape=',delta.shape,'w2.T.shape=',w2.T.shape,'X.shape=',self.X.shape
        delta2=np.dot(delta, w2.T)*self.d_act_func(self.X)
        return delta2

    def backward_prop(self, delta):
        a21=np.c_[[1], self.X]
        d2=np.dot(delta.T, a21).T
        #print  'input_dim=',self.input_dim,'output_dim=',self.output_dim,'lyr_weight.shape=',self.lyr_weight.shape,'d2.shape=',d2.shape,'alpha.shape=',self.alpha
        self.lyr_weight-=self.alpha*(d2)

    def weight_to_file(self, id):
        file2write=open('weight_lyr_'+str(id), 'wb')
        self.lyr_weight.tofile(file2write)
        file2write.close()
        

class layer_param:
    def __init__(self, input_dim, output_dim, act_func, act_deri, alpha):
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.act_func=act_func
        self.act_deri=act_deri
        self.alpha=alpha

class nnetwork:
    def __init__(self, layer_num, layer_param, input_dim, output_dim, epochs, threshold):
        time1=time.time()
        self.threshold=threshold
        self.epochs=epochs
        self.layer_num=layer_num
        self.layers=[]
        for i in range(layer_num):
            self.layers.append(layer(layer_param[i].input_dim, layer_param[i].output_dim, layer_param[i].act_func, layer_param[i].act_deri, layer_param[i].alpha, i))
        self.sample_size=input_dim
        self.output_dim=output_dim
        print ('init use ',time.time()-time1)
        
    def train(self, X, Y, sample_num, Xv, Yv, v_num):
        self.train_num=int(sample_num*9/10)
        self.validate_num=sample_num-self.train_num
        self.trainX=X[0:self.train_num,:]
        self.trainY=Y[0:self.train_num,:]
        self.validateX=X[self.train_num:sample_num,:]
        self.validateY=Y[self.train_num:sample_num,:]
        #self.validate_num=v_num
        #self.validateX=Xv
        #self.validateY=Yv
        print ('----training start----')
        for k in range(self.epochs):
            time1=time.time()
            for i in range(self.train_num):
                X=self.trainX[i]
                X=X.reshape(1, self.sample_size)
                y=self.trainY[i]
                yy=np.zeros((1, self.output_dim))
                yy[0, int(y[0])]=1.0
                for j in range(self.layer_num):
                    #print 'forward prop ',j,' X.shape=',X.shape
                    X=self.layers[j].forward_prop(X)
                error=X-yy
                delta=error
                for j in range(self.layer_num):
                    #print 'backward prop ',j,' delta.shape=',delta.shape
                    # self.layers[self.layer_num-j-1].backward_prop(delta)

                    a21=np.c_[[1], self.layers[self.layer_num-j-1].X]
                    d2=np.dot(delta.T, a21).T
                    #print  'input_dim=',self.layers[self.layer_num-j-1].input_dim,'output_dim=',self.layers[self.layer_num-j-1].output_dim,'lyr_weight.shape=',self.layers[self.layer_num-j-1].lyr_weight.shape,'d2.shape=',d2.shape,'alpha.shape=',self.layers[self.layer_num-j-1].alpha
                    self.layers[self.layer_num-j-1].lyr_weight-=self.layers[self.layer_num-j-1].alpha*(d2)

                    if j != self.layer_num-1:
                        delta=self.layers[self.layer_num-j-1].calc_delta(delta)
            print ('training',k,' use ',time.time()-time1)
            time1=time.time()
            accuracy=self.validate()
            print ('training', k,'done, accuracy=',accuracy)
            print ('validate',k,' use ',time.time()-time1)
            if accuracy > self.threshold:
                break
        print ('----training finish----')

    def train_stochastic(self, X, y):
        X=X.reshape(1, self.sample_size)
        #y=input("prompt")
        yy=np.zeros((1, self.output_dim))
        yy[0, y]=1.0
        for j in range(self.layer_num):
            #print 'forward prop ',j,' X.shape=',X.shape
            X=self.layers[j].forward_prop(X)
        index=np.argmax(X)
        error=X-yy
        delta=error
        for j in range(self.layer_num):
            #print 'backward prop ',j,' delta.shape=',delta.shape
            # self.layers[self.layer_num-j-1].backward_prop(delta)

            a21=np.c_[[1], self.layers[self.layer_num-j-1].X]
            d2=np.dot(delta.T, a21).T
            #print  'input_dim=',self.layers[self.layer_num-j-1].input_dim,'output_dim=',self.layers[self.layer_num-j-1].output_dim,'lyr_weight.shape=',self.layers[self.layer_num-j-1].lyr_weight.shape,'d2.shape=',d2.shape,'alpha.shape=',self.layers[self.layer_num-j-1].alpha
            self.layers[self.layer_num-j-1].lyr_weight-=self.layers[self.layer_num-j-1].alpha*(d2)

            if j != self.layer_num-1:
                delta=self.layers[self.layer_num-j-1].calc_delta(delta)
        #print('stochastic training, y='+str(y))
        self.dump2file()
        return y, index
        
    def reinit_weight_rand(self):
        for i in range(self.layer_num):
            self.layers[i].reinit_weight_rand()

    def validate(self):
        right=0
        wrong=0
        for i in range(self.validate_num):
            X=self.validateX[i]
            X=X.reshape(1, self.sample_size)
            y=self.validateY[i]
            yy=np.zeros((1, self.output_dim))
            yy[0, int(y[0])]=1.0
            for j in range(self.layer_num):
                X=self.layers[j].forward_prop(X)
            index=np.argmax(X)
            if index==y:
                right+=1
            else:
                wrong+=1
                #print('pred=',str(index),' actural=',str(y))
        accuracy=right/self.validate_num
        return accuracy

    def predict_validate(self, Xp, Yp, sample_num):
        right_num=0
        wrong_num=0
        for i in range(sample_num):
            X=Xp[i]
            X=X.reshape(1, self.sample_size)
            y=Yp[i]
            yy=np.zeros((1, self.output_dim))
            yy[0, int(y[0])]=1.0
            for j in range(self.layer_num):
                    #print 'forward prop ',j,' X.shape=',X.shape
                    X=self.layers[j].forward_prop(X)
            pred_y=np.argmax(X)
            if pred_y==y:
                right_num+=1
            else:
                wrong_num+=1
        return right_num, wrong_num

    def predict(self, X):
        X=X.reshape(1, self.sample_size)
        for j in range(self.layer_num):
            X=self.layers[j].forward_prop(X)
            pred_y=np.argmax(X)
        return pred_y

    def dump2file(self):
        for j in range(self.layer_num):
            self.layers[j].weight_to_file(j)



#---------Convolutional Neural Network
#----not finished, switch to tensorflow....
class conv_param:
    def __init__(self, width, height, depth):
        self.width=width
        self.height=height
        self.depth=depth

class conv_layer:
    def __init__(self, conv_dim, stride, zero_padding):
        self.cvwidth=conv_dim.width
        self.cvheight=conv_dim.height
        self.cvdepth=conv_dim.depth
        self.stide=stride
        self.padding=zero_padding
        weight_file='conv_weight'+str(self.x)+'x'+str(self.y)+'x'+str(self.z)
        if(os.path.isfile(weight_file)):
            self.weights=np.fromfile(weight_file)
            #self.weights=self.weights.reshape(self.x,self.y, self.z)
            print('init conv weights: with existing weights in file: '+weight_file)
        else:
            self.weights=np.zeros((self.x, self.y, self.z))
            print('init conv weights',str(self.x), 'x', str(self.y), 'x', str(self.z),' with random numbers')

    def forward(self, X, Y):
        #add padding
        #loop epoch
            #loop stride
                #train
        w=X.shape[0]
        h=X.shape[1]
        d=X.shape[2]
        Xt=np.zeros((x+self.padding*2, h+self.padding*2, d))
        Xt[self.padding:w+self.padding, self.padding:h+self.padding, d]=X
        cv_wloop=floor((w-self.cvwidth)/self.stride+1)
        cv_hloop=fllor((h-self.cvheight)/self.stride+1)
        a1=np.zeros((cv_wloop, cv_hloop, self.cvdepth))
        for j in range(cv_wloop):
            for k in range(cv_wloop):
                X1=Xt[self.cvwidth+j*stride, self.cvheight+k*stride, l]
                X1=X1.reshape(self.cvwidth,self.cvheight)
                for l in range(cv_wloop):
                    a=np.dot(X1, self.weights[l])
                    a1[j,k,l]=np.sum(a)
        return a1

class cnn:
    def __init__(self, epochs, threshold, alpha, input_dim,conv_dim, depth, stride, zero_padding, relu_param, pool_param, fc_param, output_dim):
        time1=time.time()
        self.threshold=threshold
        self.epochs=epochs
        self.layer_num=layer_num
        self.alpha=alpha
        self.width=input_dim.width
        self.height=input_dim.height
        self.depth=input_dim.depth

        self.convlayers=conv_layer(self, conv_dim, stride, zero_padding)
        self.relulayers=np.zeros((relu_param.width, relu_param.height, relu_param.depth))
        self.poollayers=np.zeros((pool_param.width, pool_param.height, pool_param.depth))
        self.fclayers=np.zeros((fc_param.width, fc_param.height, fc_param.depth))
        self.output_dim=output_dim
        print ('init use ',time.time()-time1)

    def train(self, X, Y, sample_num, Xv, Yv, v_num):
        self.train_num=int(sample_num*9/10)
        self.validate_num=sample_num-self.train_num
        self.trainX=X[0:self.train_num,:]
        self.trainY=Y[0:self.train_num,:]
        self.validateX=X[self.train_num:sample_num,:]
        self.validateY=Y[self.train_num:sample_num,:]
        print ('----training start----')
        for k in range(self.epochs):
            time1=time.time()
            for i in range(self.train_num):
                X=self.trainX[i]
                X=X.reshape(1, self.sample_size)
                y=self.trainY[i]
                yy=np.zeros((1, self.output_dim))
                yy[0, int(y[0])]=1.0
                for j in range(self.layer_num):
                    #print 'forward prop ',j,' X.shape=',X.shape
                    X=self.layers[j].forward_prop(X)
                error=X-yy
                delta=error
                for j in range(self.layer_num):
                    #print 'backward prop ',j,' delta.shape=',delta.shape
                    # self.layers[self.layer_num-j-1].backward_prop(delta)

                    a21=np.c_[[1], self.layers[self.layer_num-j-1].X]
                    d2=np.dot(delta.T, a21).T
                    #print  'input_dim=',self.layers[self.layer_num-j-1].input_dim,'output_dim=',self.layers[self.layer_num-j-1].output_dim,'lyr_weight.shape=',self.layers[self.layer_num-j-1].lyr_weight.shape,'d2.shape=',d2.shape,'alpha.shape=',self.layers[self.layer_num-j-1].alpha
                    self.layers[self.layer_num-j-1].lyr_weight-=self.layers[self.layer_num-j-1].alpha*(d2)

                    if j != self.layer_num-1:
                        delta=self.layers[self.layer_num-j-1].calc_delta(delta)
            print ('training',k,' use ',time.time()-time1)
            time1=time.time()
            accuracy=self.validate()
            print ('training', k,'done, accuracy=',accuracy)
            print ('validate',k,' use ',time.time()-time1)
            if accuracy > self.threshold:
                break
        print ('----training finish----')

    def train_stochastic(self, X, y):
        X=X.reshape(1, self.sample_size)
        #y=input("prompt")
        yy=np.zeros((1, self.output_dim))
        yy[0, y]=1.0
        for j in range(self.layer_num):
            #print 'forward prop ',j,' X.shape=',X.shape
            X=self.layers[j].forward_prop(X)
        index=np.argmax(X)
        error=X-yy
        delta=error
        for j in range(self.layer_num):
            #print 'backward prop ',j,' delta.shape=',delta.shape
            # self.layers[self.layer_num-j-1].backward_prop(delta)

            a21=np.c_[[1], self.layers[self.layer_num-j-1].X]
            d2=np.dot(delta.T, a21).T
            #print  'input_dim=',self.layers[self.layer_num-j-1].input_dim,'output_dim=',self.layers[self.layer_num-j-1].output_dim,'lyr_weight.shape=',self.layers[self.layer_num-j-1].lyr_weight.shape,'d2.shape=',d2.shape,'alpha.shape=',self.layers[self.layer_num-j-1].alpha
            self.layers[self.layer_num-j-1].lyr_weight-=self.layers[self.layer_num-j-1].alpha*(d2)

            if j != self.layer_num-1:
                delta=self.layers[self.layer_num-j-1].calc_delta(delta)
        #print('stochastic training, y='+str(y))
        self.dump2file()
        return y, index
        
    def reinit_weight_rand(self):
        for i in range(self.layer_num):
            self.layers[i].reinit_weight_rand()

    def validate(self):
        right=0
        wrong=0
        for i in range(self.validate_num):
            X=self.validateX[i]
            X=X.reshape(1, self.sample_size)
            y=self.validateY[i]
            yy=np.zeros((1, self.output_dim))
            yy[0, int(y[0])]=1.0
            for j in range(self.layer_num):
                X=self.layers[j].forward_prop(X)
            index=np.argmax(X)
            if index==y:
                right+=1
            else:
                wrong+=1
                #print('pred=',str(index),' actural=',str(y))
        accuracy=right/self.validate_num
        return accuracy

    def predict_validate(self, Xp, Yp, sample_num):
        right_num=0
        wrong_num=0
        for i in range(sample_num):
            X=Xp[i]
            X=X.reshape(1, self.sample_size)
            y=Yp[i]
            yy=np.zeros((1, self.output_dim))
            yy[0, int(y[0])]=1.0
            for j in range(self.layer_num):
                    #print 'forward prop ',j,' X.shape=',X.shape
                    X=self.layers[j].forward_prop(X)
            pred_y=np.argmax(X)
            if pred_y==y:
                right_num+=1
            else:
                wrong_num+=1
        return right_num, wrong_num

    def predict(self, X):
        X=X.reshape(1, self.sample_size)
        for j in range(self.layer_num):
            X=self.layers[j].forward_prop(X)
            pred_y=np.argmax(X)
        return pred_y

    def dump2file(self):
        for j in range(self.layer_num):
            self.layers[j].weight_to_file(j)















