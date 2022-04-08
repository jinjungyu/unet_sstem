from tensorflow.keras.utils import Sequence
import os
import numpy as np

class DataGenerator(Sequence):
    def __init__(self, data_dir,batch_size=1,
                 shape=(512,512,1),rescale=True,
                 shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shape = shape
        self.rescale = rescale
        self.shuffle = shuffle

        list_data = os.listdir(self.data_dir)
        list_X = [file for file in list_data if file.startswith('input')]
        list_Y = [file for file in list_data if file.startswith('label')]
        list_X.sort()
        list_Y.sort()

        self.list_X = list_X
        self.list_Y = list_Y
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.list_X)/self.batch_size))

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        X_batch = np.zeros(shape=(self.batch_size,*self.shape),dtype=np.uint8)
        Y_batch = np.zeros(shape=(self.batch_size,*self.shape),dtype=np.uint8)

        for i, idx in enumerate(indices):
            x = np.load(os.path.join(self.data_dir, self.list_X[idx]))
            y = np.load(os.path.join(self.data_dir, self.list_Y[idx]))
            # expand dimension for grayscale image
            if x.ndim == 2:
                x = np.expand_dims(x,axis=-1)
            if y.ndim == 2:
                y = np.expand_dims(y,axis=-1)
            X_batch[i] = x
            Y_batch[i] = y

        if self.rescale == True:
            X_batch = X_batch / 255

        return X_batch, Y_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_X))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

if __name__=="__main__":
    import matplotlib.pyplot as plt

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR,"dataset")

    train_path = os.path.join(DATA_DIR,"train")
    val_path = os.path.join(DATA_DIR,"val")
    test_path = os.path.join(DATA_DIR,"test")

    train_gen = DataGenerator(data_dir=train_path)
    x,y = train_gen.__getitem__(0)
    print(x.shape,x.dtype)
    print(y.shape,y.dtype)

    for i in range(len(x)):
        plt.subplot(121)
        plt.title("x")
        plt.imshow(x[i],cmap="gray")
        plt.subplot(122)
        plt.title("y")
        plt.imshow(y[i],cmap="gray")
        plt.tight_layout()
        plt.show()

        plt.subplot(121)
        plt.title("x")
        plt.hist(x[i].flatten(),bins=20)
        plt.subplot(122)
        plt.title("y")
        plt.hist(y[i].flatten(),bins=20)
        plt.tight_layout()
        plt.show()
