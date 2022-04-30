import os
import matplotlib.pyplot as plt
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(ROOT_DIR,"result")

result_train_dir = os.path.join(RESULT_DIR,"train")
result_val_dir = os.path.join(RESULT_DIR,"val")
result_test_dir = os.path.join(RESULT_DIR,"test")

os.chdir(result_train_dir)
list_file = os.listdir(result_train_dir)
list_input = sorted([file for file in list_file if file.startswith('input')])
list_label = sorted([file for file in list_file if file.startswith('label')])
list_output = sorted([file for file in list_file if file.startswith('output')])
for i in range(3):
    fig, ax = plt.subplots(1,3,figsize=(9,6))
    input_ = Image.open(list_input[i])
    label_ = Image.open(list_label[i])
    output_ = Image.open(list_output[i])

    ax[0].imshow(input_,cmap="gray")
    ax[1].imshow(label_,cmap="gray")
    ax[2].imshow(output_,cmap="gray")
    ax[0].set_title(list_input[i])
    ax[1].set_title(list_label[i])
    ax[2].set_title(list_output[i])
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()

    fig.suptitle("Train Data")
    fig.tight_layout()
    plt.show()

    input_.close()
    label_.close()
    output_.close()


os.chdir(result_val_dir)
list_file = os.listdir(result_val_dir)
list_input = sorted([file for file in list_file if file.startswith('input')])
list_label = sorted([file for file in list_file if file.startswith('label')])
list_output = sorted([file for file in list_file if file.startswith('output')])
for i in range(len(list_input)):
    fig, ax = plt.subplots(1,3,figsize=(9,6))
    input_ = Image.open(list_input[i])
    label_ = Image.open(list_label[i])
    output_ = Image.open(list_output[i])

    ax[0].imshow(input_,cmap="gray")
    ax[1].imshow(label_,cmap="gray")
    ax[2].imshow(output_,cmap="gray")
    ax[0].set_title(list_input[i])
    ax[1].set_title(list_label[i])
    ax[2].set_title(list_output[i])
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    
    fig.suptitle("Validation Data")
    fig.tight_layout()
    plt.show()

    input_.close()
    label_.close()
    output_.close()

os.chdir(result_test_dir)
list_file = os.listdir(result_test_dir)
list_input = sorted([file for file in list_file if file.startswith('input')])
list_label = sorted([file for file in list_file if file.startswith('label')])
list_output = sorted([file for file in list_file if file.startswith('output')])
for i in range(len(list_input)):
    fig, ax = plt.subplots(1,3,figsize=(9,6))
    input_ = Image.open(list_input[i])
    label_ = Image.open(list_label[i])
    output_ = Image.open(list_output[i])

    ax[0].imshow(input_,cmap="gray")
    ax[1].imshow(label_,cmap="gray")
    ax[2].imshow(output_,cmap="gray")
    ax[0].set_title(list_input[i])
    ax[1].set_title(list_label[i])
    ax[2].set_title(list_output[i])
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    
    fig.suptitle("Test Data")
    fig.tight_layout()
    plt.show()

    input_.close()
    label_.close()
    output_.close()