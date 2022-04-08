# Semantic Segmentation Practice 2
ssTem Dataset : dataset/train-volume.tif,train-labels.tif,test-volume.tif

## How to Setup and Train
1. clone repository
```
git clone https://github.com/realJun9u/unet_sstem.git
```
2. Setup Dataset (Seperate Datasets in ./dataset)
```
python setup.py
```
3. Train Model
```
python train.py
```
4. Test Model
```
python evaluate.py
```
5. Analyze Result
```
tensorboard --log_dir=logs --host={host} --port={port}
