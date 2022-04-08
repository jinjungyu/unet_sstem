import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import DataGenerator

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR,"dataset")

train_dir = os.path.join(DATA_DIR,"train")
val_dir = os.path.join(DATA_DIR,"val")
test_dir = os.path.join(DATA_DIR,"test")

train_generator = DataGenerator(data_dir=train_dir)
val_generator = DataGenerator(data_dir=val_dir)
test_generator = DataGenerator(data_dir=test_dir)

model_path = os.path.join(ROOT_DIR,'model_best.h5')

model = tf.keras.models.load_model(model_path)
# # Evaludate model
# preds_train = model.predict(train_generator,use_multiprocessing=True)
# preds_train_mask = (preds_train>0.5).astype(np.uint8)
# preds_val = model.predict(val_generator,use_multiprocessing=True)
# preds_val_mask = (preds_val>0.5).astype(np.uint8)
# preds_test = model.predict(test_generator,use_multiprocessing=True)
# preds_test_mask = (preds_test>0.5).astype(np.uint8)

# Train Prediction and Ground Truth
for idx in range(train_generator.__len__()):
    X_batch, Y_batch = train_generator.__getitem__(idx)
    preds_train = model.predict(train_generator,use_multiprocessing=True)
    preds_train_mask = (preds_train>0.5).astype(np.uint8)
    for i in range(X_batch.shape[0]):
        plt.figure(figsize=(9,3))
        plt.subtitle("Train Image Visualize")
        plt.subplot(131)
        plt.title("Train Image")
        plt.imshow(X_batch[i],cmap="gray")
        plt.subplot(132)
        plt.title("Train Prediction")
        plt.imshow(preds_train_mask[i],cmap="gray")
        plt.subplot(133)
        plt.title("Ground Truth")
        plt.imshow(Y_batch[i].astype(np.uint8),cmap="gray")
        plt.tight_layout()
        plt.show()

# Visualize Test Prediction
for idx in range(val_generator.__len__()):
    X_batch, Y_batch = val_generator.__getitem__(idx)
    preds_val = model.predict(val_generator,use_multiprocessing=True)
    preds_val_mask = (preds_val>0.5).astype(np.uint8)
    for i in range(X_batch.shape[0]):
        plt.figure(figsize=(9,3))
        plt.subtitle("Validation Image Visualize")
        plt.subplot(131)
        plt.title("val Image")
        plt.imshow(X_batch[i],cmap="gray")
        plt.subplot(132)
        plt.title("val Prediction")
        plt.imshow(preds_val_mask[i],cmap="gray")
        plt.subplot(133)
        plt.title("Ground Truth")
        plt.imshow(Y_batch[i].astype(np.uint8),cmap="gray")
        plt.tight_layout()
        plt.show()

for idx in range(test_generator.__len__()):
    X_batch, Y_batch = test_generator.__getitem__(idx)
    preds_test = model.predict(test_generator,use_multiprocessing=True)
    preds_test_mask = (preds_test>0.5).astype(np.uint8)
    for i in range(X_batch.shape[0]):
        plt.figure(figsize=(9,3))
        plt.subtitle("Test Image Visualize")
        plt.subplot(131)
        plt.title("test Image")
        plt.imshow(X_batch[i],cmap="gray")
        plt.subplot(132)
        plt.title("test Prediction")
        plt.imshow(preds_test_mask[i],cmap="gray")
        plt.subplot(133)
        plt.title("Ground Truth")
        plt.imshow(Y_batch[i].astype(np.uint8),cmap="gray")
        plt.tight_layout()
        plt.show()