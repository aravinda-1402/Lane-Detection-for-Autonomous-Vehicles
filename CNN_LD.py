#image input size: 80 x 160 x 3 (RGB)
#image output size: 80 x 160 x 1 (just the G channel with a re-drawn lane)

#****import statements****#
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Activation, Dropout, UnSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

#****loading data****#
train_images=pickle.load(open("/home/ghosh/Documents/full_CNN_train.p","rb"))
labels=pickle.load(open("/home/ghosh/Documents/full_CNN_labels.p","rb"))
train_images=np.array(train_images)
labels=np.array(labels)
labels=labels/255 #normalize labels

#****shuffle and split the training set****#
train_images,labels=shuffle(train_images, labels)
X_train, X_val, Y_train, Y_val= train_test_split(train_images,labels, test_size=0.1)

#****initializations****#
batch_size=128
pool_size= (2,2)
epochs=10
input_shape=X_train.shape[1:]

#*****NEURAL NET ARCHITECTURE*****#

model= Sequential()

#**batch normalization**#
model.add((input_shape=input_shape))

#**encoding layers**#
model.add(Conv2D(8, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv1'))
model.add(Conv2D(16, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv2'))
model.add(MaxPooling2D(pool_size=pool_size, name='Pool1'))
model.add(Conv2D(16, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv3'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv4'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv5'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=pool_size, name='Pool2'))
model.add(Conv2D(64, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv6'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv7'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=pool_size, name='Pool3'))

#**decoding layers**#
model.add(UpSampling2D(size=pool_size, name='UpSample1'))
model.add(Conv2DTranspose(64, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv1'))
model.add(Dropout(0.2))
model.add(Conv2DTranspose(64, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv2'))
model.add(Dropout(0.2))
model.add(UpSampling2D(size=pool_size, name='UpSample2'))
model.add(Conv2DTranspose(32, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv3'))
model.add(Dropout(0.2))
model.add(Conv2DTranspose(32, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv4'))
model.add(Dropout(0.2))
model.add(Conv2DTranspose(16, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv5'))
model.add(Dropout(0.2))
model.add(UpSampling2D(size=pool_size,name='UpSample3'))
model.add(Conv2DTranspose(16, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv6'))
model.add(Dropout(0.2))
model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final'))

#****data augmentation****#
datagen = ImageDataGenerator(channel_shift_range=0.2)
datagen.fit(X_train)

#****compile and train models****#
model.compile(optimizer='Adam',loss='mean_squared_error')
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size,
epochs=epochs, verbose=1, validation_data=(X_val, y_val))

#****freeze layers****#
model.trainable = False
model.compile(optimizer='Adam', loss='mean_squared_error')

#****save weights****#
model.save('/home/ghosh/Documents/full_CNN_model.h5')

model.summary()







