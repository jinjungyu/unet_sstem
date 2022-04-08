# 루트 디렉토리에 setup.py, dataset 폴더가 위치
# dataset 폴더 밑에는 train-volume.tif, train-labels.tif가 위치
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR,"dataset")
IMG_PATH = os.path.join(DATA_DIR,"train-volume.tif")
LABEL_PATH = os.path.join(DATA_DIR,"train-labels.tif")

input_total = Image.open(IMG_PATH)
label_total = Image.open(LABEL_PATH)

img_width,img_height = input_total.size
num_frame = input_total.n_frames

# 8 1 1 로 분할
num_train = int(num_frame * 0.8)
num_val = int(num_frame * 0.1)
num_test = int(num_frame * 0.1) # for prediction

X_train = np.zeros((num_train,img_height,img_width),dtype=np.uint8)
Y_train = np.zeros((num_train,img_height,img_width),dtype=np.uint8)
X_val = np.zeros((num_val,img_height,img_width),dtype=np.uint8)
Y_val = np.zeros((num_val,img_height,img_width),dtype=np.uint8)
X_test = np.zeros((num_test,img_height,img_width),dtype=np.uint8)
Y_test = np.zeros((num_test,img_height,img_width),dtype=np.uint8)

TRAIN_DIR = os.path.join(DATA_DIR,"train")
VAL_DIR = os.path.join(DATA_DIR,"val")
TEST_DIR = os.path.join(DATA_DIR,"test")

os.makedirs(TRAIN_DIR,exist_ok=True)
os.makedirs(VAL_DIR,exist_ok=True)
os.makedirs(TEST_DIR,exist_ok=True)

# Shuffle Dataset
indices = np.arange(num_frame)
np.random.shuffle(indices)

offset = 0
for i in range(num_train):
    input_total.seek(indices[offset+i])
    label_total.seek(indices[offset+i])

    input_ = np.asarray(input_total)
    label_ = (np.asarray(label_total)>1).astype(np.uint8)

    np.save(os.path.join(TRAIN_DIR,f'input_{i:03d}.npy'),input_)
    np.save(os.path.join(TRAIN_DIR,f'label_{i:03d}.npy'),label_)

offset += num_train
for i in range(num_val):
    input_total.seek(indices[offset+i])
    label_total.seek(indices[offset+i])

    input_ = np.asarray(input_total)
    label_ = (np.asarray(label_total)>1).astype(np.uint8)

    np.save(os.path.join(VAL_DIR,f'input_{i:03d}.npy'),input_)
    np.save(os.path.join(VAL_DIR,f'label_{i:03d}.npy'),label_)

offset += num_val
for i in range(num_test):
    input_total.seek(indices[offset+i])
    label_total.seek(indices[offset+i])

    input_ = np.asarray(input_total)
    label_ = (np.asarray(label_total)>1).astype(np.uint8)

    np.save(os.path.join(TEST_DIR,f'input_{i:03d}.npy'),input_)
    np.save(os.path.join(TEST_DIR,f'label_{i:03d}.npy'),label_)

# plt.subplot(121)
# plt.title("input")
# plt.imshow(input_,cmap="gray")
# plt.subplot(122)
# plt.title("label")
# plt.imshow(label_,cmap="gray")
# plt.tight_layout()
# plt.show()

# plt.subplot(121)
# plt.title("input")
# plt.hist(input_.flatten(),bins=20)
# plt.subplot(122)
# plt.title("label")
# plt.hist(label_.flatten(),bins=20)
# plt.tight_layout()
# plt.show()
