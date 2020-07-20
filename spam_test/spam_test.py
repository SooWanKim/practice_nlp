import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv", filename="spam.csv"
)
data = pd.read_csv("spam.csv", encoding="latin1")

print("총 샘플의 수 :", len(data))

print(data[:5])

del data["Unnamed: 2"]
del data["Unnamed: 3"]
del data["Unnamed: 4"]
data["v1"] = data["v1"].replace(["ham", "spam"], [0, 1])
print(data[:5])
data.info()

data.isnull().values.any()
data["v2"].nunique(), data["v1"].nunique()
data.drop_duplicates(subset=["v2"], inplace=True)
print("총 샘플의 수 :", len(data))
data["v1"].value_counts().plot(kind="bar")

# df = pd.DataFrame({'v2':["this message is about computer graphics and 3D modeling"]})
# data = data.append(df, ignore_index=True)

X_data = data["v2"]
y_data = data["v1"]

print("메일 본문의 개수: {}".format(len(X_data)))
print("레이블의 개수: {}".format(len(y_data)))

print(X_data[:5:-1])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data)
sequences = tokenizer.texts_to_sequences(X_data)
print(sequences)

word_to_index = tokenizer.word_index
print(word_to_index)

vocab_size = len(word_to_index) + 1
print("단어 집합의 크기: {}".format((vocab_size)))

n_of_train = int(len(sequences) * 0.8)
n_of_test = int(len(sequences) - n_of_train)
print("훈련 데이터의 개수 :", n_of_train)
print("테스트 데이터의 개수:", n_of_test)

X_data = sequences
max_len = 189
data = pad_sequences(X_data, maxlen=max_len)
print("훈련 데이터의 크기(shape): ", data.shape)

X_test = data[n_of_train:]  # X_data 데이터 중에서 뒤의 1034개의 데이터만 저장
y_test = np.array(y_data[n_of_train:])  # y_data 데이터 중에서 뒤의 1034개의 데이터만 저장
X_train = data[:n_of_train]  # X_data 데이터 중에서 앞의 4135개의 데이터만 저장
y_train = np.array(y_data[:n_of_train])  # y_data 데이터 중에서 앞의 4135개의 데이터만 저장

model = Sequential()
model.add(Embedding(vocab_size, 32))  # 임베딩 벡터의 차원은 32
model.add(SimpleRNN(32))  # RNN 셀의 hidden_size는 32
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(X_train, y_train, epochs=4, batch_size=64, validation_split=0.2)

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
