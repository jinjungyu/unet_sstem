import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Conv2D,MaxPool2D
from tensorflow.keras.layers import Conv2DTranspose,concatenate
from tensorflow.keras.layers import BatchNormalization

class Unet(Model):
    def __init__(self,input_shape):
        super(Unet,self).__init__()
        self.img_height,self.img_width,self.img_channels=input_shape
        
        # Conv2D + BatchNormalization2D
        def CBN2D(filters,k=3,s=1,activation="relu",padding="same",kernel_initializer='he_normal'):
            layers = Sequential([Conv2D(filters,kernel_size=(k,k),
                                                     strides=s,activation=activation,
                                                     kernel_initializer=kernel_initializer,
                                                     padding=padding),
                                              BatchNormalization()])
            return layers
        
        # 512 512 1
        # Encoder
        inputs = Input((self.img_height,self.img_width,self.img_channels))
        enc1_1 = CBN2D(filters=64)(inputs)
        enc1_2 = CBN2D(filters=64)(enc1_1)
        pool1 = MaxPool2D((2,2))(enc1_2)
        # 256 256 64
        enc2_1 = CBN2D(filters=128)(pool1)
        enc2_2 = CBN2D(filters=128)(enc2_1)
        pool2 = MaxPool2D((2,2))(enc2_2)
        # 128 128 128
        enc3_1 = CBN2D(filters=256)(pool2)
        enc3_2 = CBN2D(filters=256)(enc3_1)
        pool3 = MaxPool2D((2,2))(enc3_2)
        # 64 64 256
        enc4_1 = CBN2D(filters=512)(pool3)
        enc4_2 = CBN2D(filters=512)(enc4_1)
        pool4 = MaxPool2D((2,2))(enc4_2)
        # 32 32 512
        enc5_1 = CBN2D(filters=1024)(pool4)
        enc5_2 = CBN2D(filters=1024)(enc5_1)
        # 32 32 1024

        # Decoder
        unpool4 = Conv2DTranspose(filters=512,kernel_size=(2,2),strides=(2,2))(enc5_2) # 64 64 512
        concat4 = concatenate([enc4_2,unpool4]) # 64 64 1024
        dec4_1 = CBN2D(filters=512)(concat4)
        dec4_2 = CBN2D(filters=512)(dec4_1)
        # 64 64 512
        unpool3 = Conv2DTranspose(filters=256,kernel_size=(2,2),strides=(2,2))(dec4_2)
        concat3 = concatenate([enc3_2,unpool3])
        dec3_1 = CBN2D(filters=256)(concat3)
        dec3_2 = CBN2D(filters=256)(dec3_1)
        # 128 128 256
        unpool2 = Conv2DTranspose(filters=128,kernel_size=(2,2),strides=(2,2))(dec3_2)
        concat2 = concatenate([enc2_2,unpool2])
        dec2_1 = CBN2D(filters=128)(concat2)
        dec2_2 = CBN2D(filters=128)(dec2_1)
        # 256 256 128
        unpool1 = Conv2DTranspose(filters=64,kernel_size=(2,2),strides=(2,2))(dec2_2)
        concat1 = concatenate([enc1_2,unpool1])
        dec1_1 = CBN2D(filters=64)(concat1)
        dec1_2 = CBN2D(filters=64)(dec1_1)
        # 512 512 64
        outputs = Conv2D(1,(1,1),activation='sigmoid',name="output")(dec1_2)
        # 512 512 1
        model = Model(inputs=inputs,outputs=outputs)

        self.model = model