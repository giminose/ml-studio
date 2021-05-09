import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.python.compiler.mlcompute import mlcompute

mlcompute.set_mlc_device(device_name='gpu')
tf.config.run_functions_eagerly(False)

max_features = 20000  # 只考慮 20000 個字彙
maxlen = 200  # 每則影評只考慮前 200 個字

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
print(f'訓練資料筆數：{len(x_train)}')
print(f'測試資料筆數：{len(x_test)}')

# 不足長度，後面補0
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)

# 可輸入不定長度的整數陣列
model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 32, input_length = maxlen))
model.add(layers.Dropout(0.25))
model.add(layers.LSTM(32))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=256, epochs=2, validation_split=0.2)
model.save('lstm/lstm.h5')

plt.figure(0)
plt.plot(history.history['accuracy'],'r')
plt.plot(history.history['val_accuracy'],'g')
plt.xticks(np.arange(0, 2, 1.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train','validation'])
plt.savefig('lstm/lstm_accuracy.png')

plt.figure(1)
plt.plot(history.history['loss'],'r')
plt.plot(history.history['val_loss'],'g')
plt.xticks(np.arange(0, 2, 1.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train','validation'])
plt.savefig('lstm/lstm_validation.png')