import torch
import numpy as np

class ToTensor(object):
    def __call__(self, data):
        X, Y = data
        X = X.transpose((2,0,1)).astype('float32')
        Y = Y.transpose((2,0,1)).astype('float32')

        data = (torch.from_numpy(X), torch.from_numpy(Y))

        return data

class Normalization(object):
    def __init__(self,mean=0.5,std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self,data):
        X,Y = data
        X = (X-self.mean)/self.std
        data = (X,Y)

        return data

class RandomFlip(object):
    def __call__(self,data):
        X,Y = data

        if np.random.random() > 0.5:
            X = np.flip(X,axis=0)
            Y = np.flip(Y,axis=0)

        if np.random.random() > 0.5:
            X = np.flip(X,axis=1)
            Y = np.flip(Y,axis=1)

        data = (X,Y)

        return data

if __name__=="__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision.transforms import Compose
    from Dataset import Dataset
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir,'dataset')
    train_dir = os.path.join(data_dir,'train')
    
    transform = Compose([Normalization(),RandomFlip(),ToTensor()])
    train_dataset = Dataset(data_dir=train_dir,transform=transform)
    data = train_dataset.__getitem__(0)
    X,Y = data
    
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(X.squeeze(),cmap="gray")
    ax[0].set_title("Input")
    ax[0].axis('off')
    
    ax[1].imshow(Y.squeeze(),cmap="gray")
    ax[1].set_title("Label")
    ax[1].axis('off')
    
    fig.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(1,2)
    ax[0].hist(X.numpy().flatten(),bins=20)
    ax[0].set_title("Input")
    
    ax[1].hist(Y.numpy().flatten(),bins=20)
    ax[1].set_title("Label")
    
    fig.tight_layout()
    plt.show()