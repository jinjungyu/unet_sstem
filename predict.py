import tensorflow as tf
import os
import numpy as np
from PIL import Image

from utils.DataGenerator import DataGenerator

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR,"dataset")
RESULT_DIR = os.path.join(ROOT_DIR,"result")

train_dir = os.path.join(DATA_DIR,"train")
val_dir = os.path.join(DATA_DIR,"val")
test_dir = os.path.join(DATA_DIR,"test")

result_train_dir = os.path.join(RESULT_DIR,"train")
result_val_dir = os.path.join(RESULT_DIR,"val")
result_test_dir = os.path.join(RESULT_DIR,"test")

os.makedirs(RESULT_DIR,exist_ok=True)
os.makedirs(result_train_dir,exist_ok=True)
os.makedirs(result_val_dir,exist_ok=True)
os.makedirs(result_test_dir,exist_ok=True)

img_width = 512
img_height = 512
img_channels = 1
batch_size = 1

train_generator = DataGenerator(data_dir=train_dir,batch_size=batch_size,shape=(img_height,img_width,img_channels))
val_generator = DataGenerator(data_dir=val_dir,batch_size=batch_size,shape=(img_height,img_width,img_channels))
test_generator = DataGenerator(data_dir=test_dir,batch_size=batch_size,shape=(img_height,img_width,img_channels))

model_path = os.path.join(ROOT_DIR,'model_best.h5')
model = tf.keras.models.load_model(model_path)

# Train Prediction and Ground Truth
for idx in range(train_generator.__len__()):
    X_batch, Y_batch = train_generator.__getitem__(idx)
    preds_train = model.predict(X_batch)
    preds_train_mask = (preds_train>0.5).astype(np.uint8)
    for i in range(X_batch.shape[0]):
        input_savepath = os.path.join(result_train_dir,"input{:03d}.png".format(1+i+(batch_size*idx)))
        label_savepath = os.path.join(result_train_dir,"label{:03d}.png".format(1+i+(batch_size*idx)))
        output_savepath = os.path.join(result_train_dir,"output{:03d}.png".format(1+i+(batch_size*idx)))
        input_ = Image.fromarray((X_batch[i]*255).astype(np.uint8))
        label_ = Image.fromarray(Y_batch[i])
        output_ = Image.fromarray(preds_train_mask[i])
        input_.save(input_savepath)
        label_.save(label_savepath)
        output_.save(input_savepath)

# Visualize Test Prediction
for idx in range(val_generator.__len__()):
    X_batch, Y_batch = val_generator.__getitem__(idx)
    preds_val = model.predict(X_batch)
    preds_val_mask = (preds_val>0.5).astype(np.uint8)
    for i in range(X_batch.shape[0]):
        input_savepath = os.path.join(result_val_dir,"input{:03d}.png".format(1+i+(batch_size*idx)))
        label_savepath = os.path.join(result_val_dir,"label{:03d}.png".format(1+i+(batch_size*idx)))
        output_savepath = os.path.join(result_val_dir,"output{:03d}.png".format(1+i+(batch_size*idx)))
        input_ = Image.fromarray((X_batch[i]*255).astype(np.uint8))
        label_ = Image.fromarray(Y_batch[i])
        output_ = Image.fromarray(preds_val_mask[i])
        input_.save(input_savepath)
        label_.save(label_savepath)
        output_.save(input_savepath)

for idx in range(test_generator.__len__()):
    X_batch, Y_batch = test_generator.__getitem__(idx)
    preds_test = model.predict(X_batch)
    preds_test_mask = (preds_test>0.5).astype(np.uint8)
    for i in range(X_batch.shape[0]):
        input_savepath = os.path.join(result_test_dir,"input{:03d}.png".format(1+i+(batch_size*idx)))
        label_savepath = os.path.join(result_test_dir,"label{:03d}.png".format(1+i+(batch_size*idx)))
        output_savepath = os.path.join(result_test_dir,"output{:03d}.png".format(1+i+(batch_size*idx)))
        input_ = Image.fromarray((X_batch[i]*255).astype(np.uint8))
        label_ = Image.fromarray(Y_batch[i])
        output_ = Image.fromarray(preds_test_mask[i])
        input_.save(input_savepath)
        label_.save(label_savepath)
        output_.save(input_savepath)