import tensorflow as tf
import os
import numpy as np
from utils.DataGenerator import DataGenerator
from models.Unet import Unet

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR,"dataset")

train_dir = os.path.join(DATA_DIR,"train")
val_dir = os.path.join(DATA_DIR,"val")
test_dir = os.path.join(DATA_DIR,"test")

train_generator = DataGenerator(data_dir=train_dir)
val_generator = DataGenerator(data_dir=val_dir)

img_width = 512
img_height = 512
img_channels = 1

unet = Unet(input_shape=(img_width,img_height,img_channels))
unet.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
unet.model.summary()
####################

model_path = os.path.join(ROOT_DIR,'model_best.h5')
# Set Callback, Checkpoint
callbacks = [
             tf.keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss',save_best_only=True),
             tf.keras.callbacks.EarlyStopping(patience=3,monitor='val_loss'),
             tf.keras.callbacks.TensorBoard(log_dir=os.path.join(ROOT_DIR,'logs'))
]
# Train model
history = unet.model.fit(train_generator,epochs=50,validation_data=val_generator,callbacks=callbacks)

