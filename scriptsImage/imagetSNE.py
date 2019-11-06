#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 23:33:28 2018

@author: macramole
"""


import h5py
from glob import glob
import os
import shutil
import imageio
import numpy as np
from multiprocessing import Pool

from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import PIL
from PIL import Image



SCALE_IMGS_TO = 64

#%% Load dataset
# 
#outputPath = "/media/macramole/stuff/Data/sauvage/"
outputPath = "/media/macramole/stuff/Data/open_images/Rose/"
imagesPath = "/media/macramole/stuff/Data/open_images/Rose/goodFiles/resized_1024/*.jpg"
h5Path = outputPath + "rose.hdf5"

if not os.path.isfile(h5Path):
    images = []
    imagePathList = glob(imagesPath)
    img = imageio.imread( imagePathList[0] )
    originalShape = img.shape
    
    
    h5pyFile = h5py.File(h5Path, "w")
#    df = h5pyFile.create_dataset("df", (len(imagePathList), originalShape[0]*originalShape[1]*originalShape[2] ), dtype='uint8')
#    df = h5pyFile.create_dataset("df", (len(imagePathList), SCALE_IMGS_TO*SCALE_IMGS_TO*3 ), dtype='uint8')
    df = h5pyFile.create_dataset("df", (len(imagePathList), SCALE_IMGS_TO, SCALE_IMGS_TO, 3 ), dtype='float32')
    df.attrs['shape'] = originalShape
    filenames = []
    filenamesErrors = []
    
    #h5pyFile.close()
    def loadImage(pathImage):
#        print(pathImage)
        img = Image.open( pathImage )
        wpercent = (SCALE_IMGS_TO / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((SCALE_IMGS_TO, hsize), PIL.Image.ANTIALIAS)
        img = np.array(img, dtype="float32")
        
#        return img.reshape((img.shape[0]*img.shape[1]*img.shape[2]))
        return (pathImage, img / 255)
    
    pool = Pool(processes = 7)
    
    
#    step = 1000
#    for i in range(0,len(imagePathList),step):
#        
#        finish = i + step
#        if finish > len(imagePathList):
#            finish = len(imagePathList)
#    images = pool.map( loadImage, glob(imagesPath)[i:finish] )
    images = pool.map( loadImage, glob(imagesPath) )
    
    for j, img in enumerate(images):
        try:
            df[j] = img[1]
            filenames.append(img[0])
        except:
            filenamesErrors.append(img[0])
    
    if len(filenamesErrors) > 0:
        print("Warning: %d errors. check filenameErrors list." % len(filenamesErrors))
else :
    h5pyFile = h5py.File(h5Path, "r")
    df = h5pyFile["df"]
    originalShape = df.attrs['shape']

#%% remove h5py

h5pyFile.close()
os.remove( h5Path )

#%% remove filenames_errors

for f in filenamesErrors:
    os.remove(f)

#%% PCA (no se usa si se usan redes)

chunk_size = df.shape[0] // 20

ipca = IncrementalPCA(n_components=300, batch_size=chunk_size, whiten=True)

for i in range(0, df.shape[0] // chunk_size ):
    print("PCA %d/%d" % (i,df.shape[0] // chunk_size))
#    ipca.partial_fit(df[i*chunk_size : (i+1)*chunk_size] / 255, check_input=False)
    ipca.partial_fit(df[i*chunk_size : (i+1)*chunk_size].reshape((chunk_size, SCALE_IMGS_TO*SCALE_IMGS_TO*3)) / 255, check_input=False)

print("Variability: %f" % ipca.explained_variance_ratio_.sum() )

dataToTSne = ipca.transform( df )


#%% Image autoencoder

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard

input_img = Input(shape=(SCALE_IMGS_TO, SCALE_IMGS_TO, 3))
filter_size = (4,4)
last_pooling = (2,2) #esto afecta el tamañao del latente (mas grande es menor el latente)

#kernel_initializer='glorot_uniform'

x = Conv2D(8, filter_size, activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, filter_size, activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, filter_size, activation='relu', padding='same')(x)
encoded = MaxPooling2D(last_pooling, padding='same')(x)

latentSpaceSize = int(encoded.shape[1])*int(encoded.shape[2]*int(encoded.shape[3]))

x = Conv2D(32, filter_size, activation='relu', padding='same')(encoded)
x = UpSampling2D(last_pooling)(x)
x = Conv2D(16, filter_size, activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, filter_size, activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, filter_size, activation='relu', padding='same')(x)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
autoencoder.compile(optimizer=optimizer, loss="mean_squared_error")

autoencoder.summary()
print("Latent space size: %d" % latentSpaceSize)

#%% train

autoencoder_history = autoencoder.fit(  df, df,
                                        epochs=50,#15 (cuando habia bocha de negras esto estaba bien),#50,
                                        batch_size=128,
                                        shuffle="batch",
                                        validation_data=(df, df)
                                         #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
                                        ) 

#loss: 0.002 (4,4) latent 512
#  0.004 latent 128


#loss: 0.01 sin negras latent 128

#loss loss: 0.0071 roses
#%% predict

dataToTSne = encoder.predict( df ).reshape(df.shape[0], latentSpaceSize ) 


#%%

tsneImages = TSNE(n_components=2).fit_transform(dataToTSne)
tsneImages= np.load(outputPath + "/roses-tsne.npy")

#%%
originalShape = (255,255)
dataToTSne= np.load("/home/macramole/Code/i3a/clases/vggFeatures_")
tsneImages = TSNE(n_components=2).fit_transform(dataToTSne)
filenames =  glob("/home/macramole/Code/i3a/clases/datasets/pokemon-images/*.png")

#%% Plot !


#plt.rcParams['figure.facecolor'] = 'black'

fig, ax = plt.subplots()
#ax.scatter( tsneImages[:,0], tsneImages[:,1], picker=5 )
ax.plot(tsneImages[:,0], tsneImages[:,1], '.', picker=5)  # 5 points tolerance
plt.axis('off')
fig.tight_layout()

figImagePreview, axImagePreview = plt.subplots()
imagePreview = axImagePreview.imshow(np.zeros(originalShape))
axImagePreview.axis("off")
figImagePreview.tight_layout()

selectedPoints = None

def line_select_callback(eclick, erelease):
    global selectedPoints
    
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    
    selectedPoints = ((tsneImages[:,0] >= x1) & (tsneImages[:,0] <= x2) & (tsneImages[:,1] >= y1) & (tsneImages[:,1] <= y2))
    print( "%d selected points" % selectedPoints.sum() )
    

def onpick(event):
    global figImagePreview, axImagePreview
    
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    points = tuple(zip(xdata[ind], ydata[ind]))
#    print('onpick points:', np.array(points[0]))
    
    index = np.argwhere(tsneImages == np.array(points[0]))[0][0]
#    img = df[index].reshape( (SCALE_IMGS_TO,SCALE_IMGS_TO,3) ) #originalShape)
#    img = df[index]
    imgPath = filenames[index]
    img = imageio.imread(imgPath)
    
    imagePreview.set_data(img)
    figImagePreview.canvas.draw()
    
    
rectSelector = RectangleSelector(ax, line_select_callback,
                                   drawtype='box', useblit=True,
                                   button=[1, 3],  # don't use middle button
                                   minspanx=5, minspany=5,
                                   spancoords='pixels',
                                   interactive=True)

rectSelector.set_active(True)

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()

#%% remove selection and redo tsne

dataToTSneFiltered = dataToTSne[ np.logical_not(selectedPoints) ]
#dataToTSneFiltered = dataToTSneFiltered[ np.logical_not(selectedPoints) ] si se quiere filtrar nuevamente
tsneImages = TSNE(n_components=2).fit_transform(dataToTSneFiltered)

#%% remove selection y hacer de vuelta red

dfFiltered = df[ np.logical_not(selectedPoints),:,:,: ]
np.save(outputPath + "dfFiltered", dfFiltered)
df = dfFiltered


#%%

h5pyFile.close()
