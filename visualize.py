import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.DataGenerator import DataGenerator

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(ROOT_DIR,"result")

result_train_dir = os.path.join(RESULT_DIR,"train")
result_val_dir = os.path.join(RESULT_DIR,"val")
result_test_dir = os.path.join(RESULT_DIR,"test")

os.chdir(result_test_dir)
list_file = os.listdir(result_train_dir)
list_input = sorted([file for file in list_file if file.startswith('input')])
list_label = sorted([file for file in list_file if file.startswith('label')])
list_output = sorted([file for file in list_file if file.startswith('output')])
for i in range(len(list_input)):
    fig, ax = plt.subplots(1,3)
    fig.tight_layout()
    ax[0].imshow(list_input[i],cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title(f"input_{i}")
    ax[1].imshow(list_label[i],cmap="gray")
    ax[1].set_axis_off()
    ax[1].set_title(f"label_{i}")
    ax[2].imshow(list_output[i],cmap="gray")
    ax[2].set_axis_off()
    ax[2].set_title(f"output_{i}")
    fig.tight_layout()
    plt.show()


