import os
import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_dir,transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        list_data = os.listdir(self.data_dir)
        
        list_input = [file for file in list_data if file.startswith("input")]
        list_label = [file for file in list_data if file.startswith("label")]
        
        self.list_input = sorted(list_input)
        self.list_label = sorted(list_label)
        
    def __len__(self):
        return len(self.list_input)
    
    def __getitem__(self, idx):
        input = np.load(os.path.join(self.data_dir,self.list_input[idx]))
        label = np.load(os.path.join(self.data_dir,self.list_label[idx]))
        
        input = input/255.0
        label = label/255.0
        
        if input.ndim == 2:
            input = np.expand_dims(input,axis=-1)
        if label.ndim == 2:
            label = np.expand_dims(label,axis=-1)
        
        data = (input, label)
        
        if self.transform:
            data = self.transform(data)
        
        return data

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir,'dataset')
    train_dir = os.path.join(data_dir,'train')
    
    train_dataset = Dataset(data_dir=train_dir)
    data = train_dataset.__getitem__(0)
    X,Y = data
    
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(X,cmap="gray")
    ax[0].set_title("Input")
    ax[0].axis('off')
    
    ax[1].imshow(Y,cmap="gray")
    ax[1].set_title("Label")
    ax[1].axis('off')
    
    fig.tight_layout()
    plt.show()
    