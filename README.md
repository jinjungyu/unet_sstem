# Semantic Segmentation - UNet 2
ssTem Dataset : dataset/train-volume.tif,train-labels.tif,test-volume.tif

## How to Setup and Train
0. clone repository
```
git clone https://github.com/realJun9u/unet_sstem.git
```
1. Install Dependancy
```
# in keras or pytorch
pip install -r requirement.txt
```
2. Setup Dataset (Seperate Datasets in ./dataset)
```
python setup.py
```
3. Train and Evaluate Model
```
python train.py
```
4. Predict Model (Create Prediction Images in result directory)
```
python predict.py
```
5. Visualize Model
```
python visualize.py
```
6. Analyze Result
```
tensorboard --logdir=logs --host={host} --port={port}
```
