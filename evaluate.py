import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.DataGenerator import DataGenerator

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR,"dataset")

train_dir = os.path.join(DATA_DIR,"train")
val_dir = os.path.join(DATA_DIR,"val")
test_dir = os.path.join(DATA_DIR,"test")

img_width = 512
img_height = 512
img_channels = 1

train_generator = DataGenerator(data_dir=train_dir,batch_size=1,shape=(img_height,img_width,img_channels))
val_generator = DataGenerator(data_dir=val_dir,batch_size=1,shape=(img_height,img_width,img_channels))
test_generator = DataGenerator(data_dir=test_dir,batch_size=1,shape=(img_height,img_width,img_channels))

model_path = os.path.join(ROOT_DIR,'model_best.h5')
model = tf.keras.models.load_model(model_path)

# Train Prediction and Ground Truth
for idx in range(train_generator.__len__()):
    X_batch, Y_batch = train_generator.__getitem__(idx)
    preds_train = model.predict(X_batch)
    preds_train_mask = (preds_train>0.5).astype(np.uint8)
    for i in range(X_batch.shape[0]):
        fig, ax =  plt.subplots(1,3)
        ax[0].set_title("Train Image")
        ax[0].imshow(X_batch[i],cmap="gray")
        ax[1].set_title("Ground Truth")
        ax[1].imshow(Y_batch[i],cmap="gray")
        ax[2].set_title("Train Prediction")
        ax[2].imshow(preds_train_mask[i],cmap="gray")
        fig.tight_layout()
        plt.show()

# Visualize Test Prediction
for idx in range(val_generator.__len__()):
    X_batch, Y_batch = val_generator.__getitem__(idx)
    preds_val = model.predict(X_batch)
    preds_val_mask = (preds_val>0.5).astype(np.uint8)
    for i in range(X_batch.shape[0]):
        fig, ax =  plt.subplots(1,3)
        ax[0].set_title("Validation Image")
        ax[0].imshow(X_batch[i],cmap="gray")
        ax[1].set_title("Ground Truth")
        ax[1].imshow(Y_batch[i],cmap="gray")
        ax[2].set_title("Validation Prediction")
        ax[2].imshow(preds_val_mask[i],cmap="gray")
        fig.tight_layout()
        plt.show()

for idx in range(test_generator.__len__()):
    X_batch, Y_batch = test_generator.__getitem__(idx)
    preds_test = model.predict(X_batch)
    preds_test_mask = (preds_test>0.5).astype(np.uint8)
    for i in range(X_batch.shape[0]):
        fig, ax =  plt.subplots(1,3)
        ax[0].set_title("Test Image")
        ax[0].imshow(X_batch[i],cmap="gray")
        ax[1].set_title("Ground Truth")
        ax[1].imshow(Y_batch[i],cmap="gray")
        ax[2].set_title("Test Prediction")
        ax[2].imshow(preds_train_mask[i],cmap="gray")
        fig.tight_layout()
        plt.show()