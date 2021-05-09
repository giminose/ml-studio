# Introduction
This is my learning record of ML with Mac M1 Tensorflow.
1. [LSTM](https://www.kaggle.com/linchs/rnn-imdb)
2. GRU
2. [Auto Encoder](https://www.kaggle.com/linchs/autoencoder)
3. [Denoising Auto Encoder](https://www.kaggle.com/linchs/denoising-ae)
4. [GAN](https://www.kaggle.com/linchs/gan-mnist)

# Install
Reference to [MacBook M1: installing TensorFlow and Jupyter Notebook](https://medium.com/gft-engineering/macbook-m1-tensorflow-on-jupyter-notebooks-6171e1f48060)
# Trouble shooting
## GPU issue
You can enable GPU training by the code below.
```python
from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='gpu')
tf.config.run_functions_eagerly(False)
```
BUT...., it will raise an error when `model.predict()` or `model.evaluate()`.
My solution is to separate the training process from predicting.   
## [LSTM error #229](https://github.com/apple/tensorflow_macos/issues/229)
conda install numpy=1.19.5

## [h5py==3.0.0 causes issues with keras model loads in tensorflow 2.1.0 #44467](https://github.com/tensorflow/tensorflow/issues/44467)
conda install hyp5==2.10.0
