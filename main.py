# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pathlib, os, random, mplcyberpunk, splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D, Activation, add, AveragePooling2D, DepthwiseConv2D, GlobalAveragePooling2D
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
        
data_bs = './images'
data_bs = pathlib.Path(data_bs)

splitfolders.ratio(data_bs, output='Imgs/', seed=1234, ratio=(0.7, 0.15, 0.15), group_prefix=None)

epochs = 30
NUM_CLASSES = 5
batch_size = 32
img_height, img_width = 300, 300
input_shape = (img_height, img_width, 3)

data_gen = ImageDataGenerator(rescale=1.0 / 255)

train_ds = data_gen.flow_from_directory('Imgs/train/', target_size=(img_height, img_width), batch_size=batch_size, class_mode='sparse', subset='training')

val_ds = data_gen.flow_from_directory('Imgs/val/', target_size=(img_height, img_width), batch_size=batch_size, class_mode='sparse', shuffle=False)

class BaseModel(tf.keras.Model):
    def __init__(self, model_name):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        
        self.C1 = Conv2D(32, (3 * 3), padding='same', input_shape=input_shape)
        self.B1 = BatchNormalization()
        self.A1 = Activation('relu')
        self.P1 = MaxPooling2D((3 * 3), 1,padding='same')

        
        self.C2 = Conv2D(32, (3 * 3), padding='same')
        self.B2 = BatchNormalization()
        self.A2 = Activation('relu')
        self.P2 = MaxPooling2D((3 * 3), 1,padding='same')
        self.Dr1 = Dropout(0.3)
        
        self.C3 = Conv2D(32, (3 * 3), padding='same')
        self.B3 = BatchNormalization()
        self.A3 = Activation('relu')
        self.P3 = MaxPooling2D(2, padding='same')
        self.Dr2 = Dropout(0.3)
        
        self.F1 = Flatten()
        self.D1 = Dense(64, activation='relu')
        self.B4 = BatchNormalization()
        self.D2 = Dense(64, activation='relu')
        self.D3 = Dense(64, activation='relu')
        self.D4 = Dense(32, activation='relu')
        self.Dr3 = Dropout(0.4)
        self.D5 = Dense(NUM_CLASSES, activation='softmax')
        

    def call(self, x):
        x = self.C1(x)
        x = self.B1(x)
        x = self.A1(x)
        x = self.P1(x)
        
        x = self.C2(x)
        x = self.B2(x)
        x = self.A2(x)
        x = self.P2(x)
        x = self.Dr1(x)
        
        x = self.C3(x)
        x = self.B3(x)
        x = self.A3(x)
        x = self.P3(x)
        x = self.Dr2(x)
        
        x = self.F1(x)
        x = self.D1(x)
        x = self.B4(x)
        x = self.D2(x)
        x = self.D3(x)
        x = self.D4(x)
        x = self.Dr3(x)
        y = self.D5(x)
        return y
    
    
    def __repr__(self):
        name = '{}_Model'.format(self.model_name)
        return name
    
    
model = BaseModel(model_name='Huang')

model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
             metrics=['sparse_categorical_accuracy'])


checkpoint_save_path = './BaseModel.weights.h5'
if os.path.exists(checkpoint_save_path + '.index'):
    model.load_weights(checkpoint_save_path)

model.save('my_model.keras')
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                save_best_only=True,
                                                save_weights_only=True)

history = model.fit(train_ds, epochs=30, batch_size=batch_size, callbacks=[cp_callback])

model.summary()

file = open('./BaseModelWeights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
    
file.close()

acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.figure(figsize=(16, 8))
plt.style.use('cyberpunk')
plt.subplot(1, 2, 1)
plt.title('BaseModel Training Acc')
plt.plot(acc, label='Training Acc')
mplcyberpunk.add_glow_effects()

plt.subplot(1, 2, 2)
plt.title('BaseModel Training Loss')
plt.plot(loss, label='Training Loss')
mplcyberpunk.add_glow_effects()


plt.show()