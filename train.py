# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:30:02 2020

@author: joshu
"""
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Adagrad, Adadelta, Adamax
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.initializers import glorot_uniform, he_normal, he_uniform
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
train_set = np.load('D:/AI/BTFE_tests/AB_CD_all/BTFE_train_set_T40v60_AB_CD_small.npy')
train_label = np.load('D:/AI/BTFE_tests/AB_CD_all/BTFE_train_labels_T40v60_AB_CD_small.npy')
val_set = np.load('D:/AI/BTFE_tests/AB_CD_all/BTFE_val_set_T40v60_AB_CD_small.npy')
val_label = np.load('D:/AI/BTFE_tests/AB_CD_all/BTFE_val_label_T40v60_AB_CD_small.npy')



model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(256, (3,3),input_shape=(32,32,1),activation='relu',data_format='channels_last',kernel_regularizer=l2(0.001), kernel_initializer='glorot_normal'),
tf.keras.layers.Conv2D(256, (3,3),activation='relu',data_format='channels_last',kernel_regularizer=l2(0.001)),
# tf.keras.layers.Dropout(0.5),
tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(1,1), padding='same', data_format='channels_last'),
tf.keras.layers.Dropout(0.1),
   
tf.keras.layers.Conv2D(128, (3,3),activation='relu',data_format='channels_last',kernel_regularizer=l2(0.001)),
tf.keras.layers.Conv2D(128, (3,3),activation='relu',data_format='channels_last',kernel_regularizer=l2(0.001), kernel_initializer='glorot_normal'),
# tf.keras.layers.Dropout(0.5),
tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(1,1), padding='same', data_format='channels_last'),
tf.keras.layers.Dropout(0.1),
   
# tf.keras.layers.Conv2D(256, (3,3),activation='relu',data_format='channels_last'),
tf.keras.layers.Conv2D(64, (3,3),activation='relu',data_format='channels_last',kernel_regularizer=l2(0.001)),
tf.keras.layers.Conv2D(64, (3,3),activation='relu',data_format='channels_last',kernel_regularizer=l2(0.001), kernel_initializer='glorot_normal'),
# tf.keras.layers.Dropout(0.5),
tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(1,1), padding='same', data_format='channels_last'),
tf.keras.layers.Dropout(0.1),
   
# tf.keras.layers.Conv2D(256, (3,3),activation='relu',data_format='channels_last'),
tf.keras.layers.Conv2D(32, (3,3),activation='relu',data_format='channels_last',kernel_regularizer=l2(0.001)),
tf.keras.layers.Conv2D(32, (3,3),activation='relu',data_format='channels_last',kernel_regularizer=l2(0.001), kernel_initializer='glorot_normal'),
# tf.keras.layers.Dropout(0.5),
tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(1,1), padding='same', data_format='channels_last'),
tf.keras.layers.Dropout(0.1),
   
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(16, activation='relu',kernel_regularizer=l2(0.001), kernel_initializer='glorot_normal'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dropout(0.1),

tf.keras.layers.Dense(1, activation='sigmoid')])


model.compile(optimizer = Adam(lr=0.000001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip=True,
    # shear_range=0.5,
    # fill_mode='nearest',
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.5,
    height_shift_range=0.5,
    )
train_datagen.fit(train_set)
train_generator = train_datagen.flow(
    train_set,train_label)


validation_datagen = ImageDataGenerator(    
    rescale = 1./255,
    horizontal_flip=True,
    # shear_range=0.5,
    # fill_mode='nearest',
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.5,
    height_shift_range=0.5,)
validation_datagen.fit(val_set)

validation_generator = validation_datagen.flow(
    val_set, val_label)

epochs = 75
batch_size = 32
DTI_spe = 177
DTI_val_steps = 180
history = model.fit(train_generator, epochs=epochs, steps_per_epoch=DTI_spe, validation_data=validation_generator,validation_steps=DTI_val_steps, shuffle=True)
# history = model.fit(train_datagen.flow(train_set,train_label), epochs=100, steps_per_epoch=1, validation_data=validation_generator)
train_loss, train_acc = model.evaluate(train_set,train_label, verbose=2)
test_loss, test_acc = model.evaluate(val_set,val_label, verbose=2)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs,acc,'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='validation accuracy')
plt.title('training and validation accuracy. simple model, ABvsCD, vs=0.2 sigmoid, binary_crossentropy,Adam(lr=0.00001), l2 kernelinit,0.001, dropout=0.1')
plt.legend(loc=0)
plt.figure()
plt.show()

plt.plot(epochs, loss, 'r', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation_loss')
plt.title('the losses')
plt.legend(loc=0)
plt.figure()
plt.show()