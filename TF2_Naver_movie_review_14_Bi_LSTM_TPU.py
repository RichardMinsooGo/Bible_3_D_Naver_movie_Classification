! git clone https://github.com/simonjisu/nsmc_study.git

from IPython.display import clear_output 
clear_output()


import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU {}'.format(tpu.cluster_spec().as_dict()['worker']))
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: {}".format(strategy.num_replicas_in_sync))

print("Tensorflow version {}".format(tf.__version__))
tf.random.set_seed(1234)
AUTO = tf.data.experimental.AUTOTUNE

# 1. Tokenizer Install & import
# Keras Tokenizer는 tensorflow 2.X 에서 기본으로 제공하는 tokenizer이며, word level tokenizer이다. 이는 별도의 설치가 필요 없다.

# 2. Copy or load raw data to Colab
max_len = 50
BATCH_SIZE  = 128
BUFFER_SIZE = 20000

import urllib3
import zipfile
import shutil
import pandas as pd

pd.set_option('display.max_colwidth', 100)
# pd.set_option('display.max_colwidth', None)

train = pd.read_csv("/content/nsmc_study/data/ratings_train.txt", sep='\t')
test = pd.read_csv("/content/nsmc_study/data/ratings_test.txt", sep='\t')

train = train.drop(columns=['id'])
test = test.drop(columns=['id'])

print(train.shape)
print(test.shape)

train_data = train.dropna() #말뭉치에서 nan 값을 제거함
test_data  = test.dropna()

# 5. Preprocess and build list

def preprocess_func(sentence):
    # 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    sentence = sentence.strip()  
    return sentence

train_data['document'] = train_data['document'].apply(preprocess_func)
test_data['document']  = test_data['document'].apply(preprocess_func)

train_data['label'] = train_data['label'].astype(int)
Label_Train = train_data["label"].to_numpy()

test_data['label'] = test_data['label'].astype(int)
Label_test = test_data["label"].to_numpy()

src_train_df = train_data['document']
src_test_df  = test_data['document']

print(src_train_df[:10])

train_sentence  = src_train_df.apply(lambda x: "<SOS> " + str(x))
test_sentence  = src_test_df.apply(lambda x: "<SOS> " + str(x))

filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'

# Define tokenizer
SRC_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)

SRC_tokenizer.fit_on_texts(train_sentence)

vocab_size = len(SRC_tokenizer.word_index) + 1

print('Encoder 단어 집합의 크기 :',vocab_size)

# 7. Tokenizer test
lines = [
  "게임 하고 싶은데 할래?",
  "나 너 좋아하는 것 같아",
  "딥 러닝 자연어 처리를 잘 하고 싶어"
]

for line in lines:
    txt_2_ids = SRC_tokenizer.texts_to_sequences([line])
    ids_2_txt = SRC_tokenizer.sequences_to_texts(txt_2_ids)
    print("Input     :", line)
    print("txt_2_ids :", txt_2_ids)
    print("ids_2_txt :", ids_2_txt[0],"\n")

# 8. Tokenize    
# 토큰화 / 정수 인코딩 / 시작 토큰과 종료 토큰 추가 / 패딩
Train_tkn_inputs = SRC_tokenizer.texts_to_sequences(train_sentence)
Test_tkn_inputs  = SRC_tokenizer.texts_to_sequences(test_sentence)

len_result = [len(s) for s in Train_tkn_inputs]

print('Maximum length of review : {}'.format(np.max(len_result)))
print('Average length of review : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

# 9. Pad sequences
# 패딩
padded_train_tkn = tf.keras.preprocessing.sequence.pad_sequences(Train_tkn_inputs,  maxlen=max_len, padding='post', truncating='post')
padded_test_tkn = tf.keras.preprocessing.sequence.pad_sequences(Test_tkn_inputs,  maxlen=max_len, padding='post', truncating='post')

# 10. Data type define
padded_train_tkn = tf.cast(padded_train_tkn, dtype=tf.int64)
padded_test_tkn = tf.cast(padded_test_tkn, dtype=tf.int64)

# 11. Check tokenized data
print('질문 데이터의 크기(shape) :', padded_train_tkn.shape)

# 0번째 샘플을 임의로 출력
print(padded_train_tkn[0])

X_train = padded_train_tkn[:125000]
y_train = Label_Train[:125000]
X_valid = padded_train_tkn[125000:]
y_valid = Label_Train[125000:]
X_test  = padded_test_tkn
y_test  = Label_test

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import load_model

from tensorflow import keras
from tensorflow.keras import layers

embedding_dim = 256
hidden_units = 128

# initialize and compile model within strategy scope
with strategy.scope():
    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a embedding_dim-dimensional vector
    x = layers.Embedding(vocab_size, embedding_dim)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(hidden_units, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(hidden_units))(x)
    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
model.summary()
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
mc = ModelCheckpoint('best_model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)
history = model.fit(X_train, y_train, epochs = 20, validation_data = (X_valid, y_valid), callbacks=[es, mc])

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'o', color='g', label='Training loss')   # 'bo'
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.plot(epochs, acc, 'o', color='g', label='Training acc')   # 'bo'
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()


def sentiment_predict(new_sentence):
    # 알파벳과 숫자를 제외하고 모두 제거 및 알파벳 소문자화
    new_sentence = preprocess_func(new_sentence)

    txt_2_ids = SRC_tokenizer.texts_to_sequences([new_sentence])

    pad_sequence = pad_sequences(txt_2_ids, maxlen=max_len) # 패딩
    score = float(loaded_model.predict(pad_sequence)) # 예측

    if(score > 0.5):
        print("A positive review with a {:.2f}% chance. ".format(score * 100))
    else:
        print("A negative review with a {:.2f}% chance. ".format((1 - score) * 100))

for idx in range(10):
    print('----'*30)
    test_input = test_sentence[20000+idx]
    print("Test sentence from datasets:\n", test_input)
    sentiment_predict(test_input)
    if(y_test[20000+idx] > 0.5):
        print("Ground truth is positive!")
    else:
        print("Ground truth is negative!")
    
    