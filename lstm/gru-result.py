import tensorflow as tf
from tensorflow import keras
import numpy as np

tf.config.run_functions_eagerly(True)

max_features = 20000  # 只考慮 20000 個字彙
maxlen = 200  # 每則影評只考慮前 200 個字

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
    num_words=max_features
)
print(f'訓練資料筆數：{len(x_train)}')
print(f'測試資料筆數：{len(x_test)}')

# 不足長度，後面補0
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = keras.models.load_model("lstm/gru.h5")

print("Evaluate")
print(model.evaluate(x_test, y_test))