from PIL import Image
import numpy as np

arr = np.load('./dataset/train/input_000.npy')
arr = np.expand_dims(arr,axis=-1)
print(arr.shape,arr.dtype,type(arr))
im = Image.fromarray(arr)
im.save('./fromarray.png')
