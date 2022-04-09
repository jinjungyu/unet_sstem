import tensorflow as tf
import os

from utils.DataGenerator import DataGenerator
from models.Unet import Unet

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR,"dataset")

train_dir = os.path.join(DATA_DIR,"train")
val_dir = os.path.join(DATA_DIR,"val")
test_dir = os.path.join(DATA_DIR,"test")

img_width = 512
img_height = 512
img_channels = 1
batch_size = 3

train_generator = DataGenerator(data_dir=train_dir,batch_size=batch_size,shape=(img_height,img_width,img_channels))
val_generator = DataGenerator(data_dir=val_dir,batch_size=batch_size,shape=(img_height,img_width,img_channels))
test_generator = DataGenerator(data_dir=test_dir,batch_size=batch_size,shape=(img_height,img_width,img_channels))

unet = Unet(input_shape=(img_width,img_height,img_channels))
unet.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
unet.model.summary()
####################

model_path = os.path.join(ROOT_DIR,'model_best.h5')
# Set Callback, Checkpoint
callbacks = [
             tf.keras.callbacks.ModelCheckpoint(model_path,monitor='val_acc',save_best_only=True),
             tf.keras.callbacks.EarlyStopping(patience=100,monitor='val_loss'),
             tf.keras.callbacks.TensorBoard(log_dir=os.path.join(ROOT_DIR,'logs'))
]
# Train model
history = unet.model.fit(train_generator,epochs=100,validation_data=val_generator,callbacks=callbacks)
results = unet.model.evaluate(test_generator)
print("Test Loss, Test Accuracy :",results)


