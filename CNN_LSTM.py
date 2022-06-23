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
from keras.layers import LSTM
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

#****NEURAL NET ARCHITECTURE****#

model= Sequential()

#**batch normalization**#
model.add(BatchNormalization(input_shape=input_shape))

#**local feature encoder**#
#INPUT:(80x160x3)
model.add(Conv2D(8, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv1')) #OUTPUT:(78x158x8)
model.add(Conv2D(16, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv2'))#OUTPUT:(76x156x16)
model.add(MaxPooling2D(pool_size=2, padding='valid', name='Pool1'))#OUTPUT:(38x78x16)

model.add(Conv2D(16, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv3'))#OUTPUT:(36x76x16)
model.add(Conv2D(32, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv4'))#OUTPUT:(34x74x32)
model.add(Conv2D(32, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv5'))#OUTPUT:(32x72x32)
model.add(MaxPooling2D(pool_size=2, padding='valid', name='Pool2'))#OUTPUT:(16x36x32)


model.add(Conv2D(64, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv6'))#OUTPUT:(14x34x64)
model.add(MaxPooling2D(pool_size=2, padding='valid', name='Pool3'))#OUTPUT:(7x17x64)

#**feature processor**#
model.add(Conv2D(64, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv7'))#OUTPUT:(5x15x64)
model.add(ConvLSTM2D(64, (1,1), input_shape=(None,5,15,64), padding='same', return_sequences=True , name='LSTM1'))#OUTPUT:(5x15x64)

model.add(Conv2D(128, (3,3), padding='valid', strides=(1,1), activation='relu', name='Conv8'))#OUTPUT:(3x13x128)
model.add(LSTM(128, (1,1), input_shape=(None,3,13,128), padding='same', return_sequences=True, name='LSTM2'))#OUTPUT:(3x13x128)

#**feature decoder**#
model.add(Conv2DTranspose(128, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv1'))#OUTPUT:(5x15x128)
model.add(Conv2DTranspose(64, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv2'))#OUTPUT:(7x17x64)

model.add(UpSampling2D(size=pool_size, name='UpSample1'))#OUTPUT:(14x34x64)
model.add(Conv2DTranspose(64, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv3'))#OUTPUT:(16x36x64)

model.add(UpSampling2D(size=pool_size, name='UpSample2'))#OUTPUT:(32x72x64)
model.add(Conv2DTranspose(32, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv4'))#OUTPUT:(34x74x32)
model.add(Conv2DTranspose(32, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv5'))#OUTPUT:(36x76x16)
model.add(Conv2DTranspose(16, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv6'))#OUTPUT:(38x78x16)
model.add(UpSampling2D(size=pool_size, name='UpSample3'))#OUTPUT:(76x156x16)

model.add(Conv2DTranspose(16, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv7'))#OUTPUT:(78x158x16)
model.add(Conv2DTranspose(1, (3,3), padding='valid', strides=(1,1), activation='relu', name='Deconv8'))#OUTPUT:(80x160x1)

#****data augmentation****#
datagen = ImageDataGenerator(channel_shift_range=0.2)
datagen.fit(X_train)

#****compile and train models****#
model.compile(optimizer='Adam',loss='mean_squared_error', metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size,
epochs=epochs, verbose=1, validation_data=(X_val, y_val))

#****freeze layers****#
model.trainable = False
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

#****save weights****#
model.save('/home/ghosh/Documents/full_CNN_model.h5')

model.summary()







