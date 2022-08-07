# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 11:03:43 2022

@author: ew
"""
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Conv1D, MaxPool1D, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.models import Sequential, Model
from keras.optimizers import Adam,Adagrad,RMSprop,Nadam,Adamax,Adadelta,SGD
from keras import losses
from mpl_toolkits.axes_grid1 import host_subplot
from keras.utils import to_categorical
import keras.backend as K
import scipy.io as sio
import matplotlib.pyplot as plt
import random
import numpy as np
import time

def plot_acc_loss(d_loss, g_loss):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    par1 = host.twinx() 
 
    # set labels
    host.set_xlabel("steps")
    host.set_ylabel("loss1")
    par1.set_ylabel("loss2")
 
    # plot curves
    p1, = host.plot(range(len(d_loss)), d_loss, label="loss1")
    p2, = par1.plot(range(len(g_loss)), g_loss, label="loss2")
 
    host.legend(loc=5)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
 
    plt.draw()
    plt.show()

def add_gaussian_noise(ori_in, noise_sigma): 
    t_in = ori_in
    h = ori_in.shape[0] 
    w = ori_in.shape[1] 
    noise = np.random.randn(h, w) * noise_sigma
    noise_sigma = np.zeros(t_in.shape, np.float32) 
    noise_out = t_in + noise     
    return noise_out


tStart = time.time()

#讀入資料

data = sio.loadmat('D:/python/map3/data/map3_data/data_orx1_47_anlog.mat')

x_train1 = data['train1']
x_test1 = data['test1']
x_train2 = data['train2']
x_test2 = data['test2']
x_train3 = data['train3']
x_test3 = data['test3']


data = sio.loadmat('D:/python/map3/data/map3_data/data_orx2_47_anlog.mat')


x_train4 = data['train1']
x_test4 = data['test1']
x_train5 = data['train2']
x_test5 = data['test2']
x_train6 = data['train3']
x_test6 = data['test3']


data = sio.loadmat('D:/python/map3/data/map3_data/data_orx3_47_anlog.mat')


x_train7 = data['train1']
x_test7 = data['test1']
x_train8 = data['train2']
x_test8 = data['test2']
x_train9 = data['train3']
x_test9 = data['test3']

x_train = np.hstack([x_train1,x_train2,x_train3,x_train4,x_train5,x_train6,x_train7,x_train8,x_train9])
x_test = np.hstack([x_test1,x_test2,x_test3,x_test4,x_test5,x_test6,x_test7,x_test8,x_test9])

y_train = data['label_train']
classes = y_train.shape[1]
csi_c = x_train.shape[1]

#選取資料

idxs_annot = range(x_train.shape[0])
random.seed(3)
idxs_annot = np.random.choice(x_train.shape[0], 2000, replace=False) 

x_train_l = x_train[idxs_annot]
y_train_l = y_train[idxs_annot]

#模型設置

def build_generator():
    
        noise = Input(shape=(100,))
        
        x = Dense(500, activation="relu")(noise)      
        x = Dense(300, activation="relu")(x)
        x = Dense(250, activation="relu")(x)
        x = Dense(100, activation="relu")(x)
        output_ = Dense(csi_c, activation="softmax")(x)
        
        model = Model(noise, output_)

        model.summary()

        return model
    
def build_discriminator():
 
        model = Sequential()     
        model.add(Dense(100, activation="relu", input_shape=(csi_c,)))
        model.add(Dense(250, activation="relu")) 
        model.add(Dense(300, activation="relu"))       
        model.add(Dense(500, activation="relu"))
        model.add(Dropout(0.3))

 
        model.summary()
 
        img = Input(shape=(csi_c,))
 
        features = model(img)
        valid = Dense(1, activation="sigmoid")(features)#sigmoid
        label = Dense(classes, activation="softmax")(features)
 
        return Model(img, [valid, label])

nb_epochs = 3000
batch_size = 100
latent_size = 100
opt = Adagrad(0.001)

discriminator = build_discriminator()
discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],#['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            optimizer=opt,
            metrics=['accuracy'])


generator = build_generator()
generator.compile(loss=['categorical_crossentropy'],
            optimizer=opt)


latent = Input(shape=(latent_size,))

fake = generator(latent)

discriminator.trainable = False

tof, aux = discriminator(fake)


combined = Model(latent, [tof, aux])
combined.compile(
        loss=['binary_crossentropy', 'categorical_crossentropy'],
        optimizer=opt       
    )

loss1=[]
loss2=[]

#開始訓練

for epochs in range(nb_epochs):
    idxs_l = range(x_train_l.shape[0])
    random.seed(4)
    idxs_l = np.random.choice(x_train_l.shape[0], batch_size, replace=False) 
    
    nu=1
    
    ix_train_l = x_train_l[idxs_l]
    iy_train_l = y_train_l[idxs_l]


    noise = np.random.normal(-1, 1, (batch_size*nu, latent_size))   
    g_labels = np.random.randint(0, classes, batch_size*nu)
    g_labels = to_categorical(g_labels.reshape(-1, 1), num_classes=classes)
    
    g_train = generator.predict(noise, verbose=0)
    
    noise_sigma = 0.3
    noise_l = add_gaussian_noise(ix_train_l, noise_sigma=noise_sigma) 
    
    train_ = np.vstack([noise_l,g_train])
    label_tf = np.array([1] * (batch_size*nu) + [0] * (batch_size*nu))
    label_c = np.vstack([iy_train_l,g_labels])
    
    d_loss = discriminator.train_on_batch(train_,[label_tf,label_c])
    
    noise = np.random.normal(-1, 1, (2*batch_size, latent_size))
    g_labels = np.random.randint(0, classes, 2*batch_size)
    g_labels = to_categorical(g_labels.reshape(-1, 1), num_classes=classes)
  
    trick = np.ones(2 * batch_size)####
    g_loss = combined.train_on_batch(noise,[trick ,g_labels])
    
    print ("%d D loss: %f , G loss: %f" % (epochs, d_loss[0], g_loss[0]))
    
    loss1.append(d_loss[0])
    loss2.append(g_loss[0])
    
    
plot_acc_loss(loss1, loss2)

t1 = discriminator.predict(x_test)
out = t1[1][:,0:classes]
sio.savemat('D:/python/map3/t1/out/mm_'+str(classes)+'.mat',dict(single_pro=out))

tEnd = time.time()    
print("It cost %f sec",(tEnd - tStart)) 















