'''
Data Engineering
'''

'''
D01. Import Libraries for Data Engineering
'''
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata

print("Tensorflow version {}".format(tf.__version__))
import random
SEED = 1234
tf.random.set_seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE

'''
D02. Import Naver Movie Review Raw Dataset from Github
'''
! git clone https://github.com/simonjisu/nsmc_study.git

from IPython.display import clear_output 
clear_output()

'''
D03. [PASS] Tokenizer Install & import
''' 
# Keras Tokenizer is a tokenizer provided by default in tensorflow 2.X and is a word level tokenizer. It does not require a separate installation.

'''
D04. Define Hyperparameters for Data Engineering
'''
max_len = 50

'''
D05. Load and modifiy to pandas dataframe
'''
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

'''
D06. [PASS] Delete duplicated data
'''

'''
D07. [PASS] Select samples
'''

'''
D08. Preprocess and build list
'''
def preprocess_func(sentence):
    # 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    sentence = sentence.strip()
    return sentence

train_data['document'] = train_data['document'].apply(preprocess_func)
test_data['document']  = test_data['document'].apply(preprocess_func)

'''
D09. Add <SOS>, <EOS> for source and target
'''

train_data['label'] = train_data['label'].astype(int)
Label_Train = train_data["label"].to_numpy()

test_data['label'] = test_data['label'].astype(int)
Label_test = test_data["label"].to_numpy()

src_train_df = train_data['document']
src_test_df  = test_data['document']

print(src_train_df[:10])

train_sentence  = src_train_df.apply(lambda x: "<SOS> " + str(x))
test_sentence  = src_test_df.apply(lambda x: "<SOS> " + str(x))

'''
D10. Define tokenizer
'''

filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'

SRC_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)

SRC_tokenizer.fit_on_texts(train_sentence)

vocab_size = len(SRC_tokenizer.word_index) + 1

print('Word set size of Encoder :',vocab_size)

'''
D11. Tokenizer test
'''

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

'''
D12. Tokenize
'''
# 토큰화 / 정수 인코딩 / 시작 토큰과 종료 토큰 추가 / 패딩
Train_tkn_inputs = SRC_tokenizer.texts_to_sequences(train_sentence)
Test_tkn_inputs  = SRC_tokenizer.texts_to_sequences(test_sentence)

'''
D13. [EDA] Explore the tokenized datasets
'''

len_result = [len(s) for s in Train_tkn_inputs]

print('Maximum length of review : {}'.format(np.max(len_result)))
print('Average length of review : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

'''
D14. Pad sequences
'''

from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_train_tkn = pad_sequences(Train_tkn_inputs,  maxlen=max_len, padding='post', truncating='post')
padded_test_tkn = pad_sequences(Test_tkn_inputs,  maxlen=max_len, padding='post', truncating='post')

'''
D15. Data type define
'''
padded_train_tkn = tf.cast(padded_train_tkn, dtype=tf.int64)
padded_test_tkn = tf.cast(padded_test_tkn, dtype=tf.int64)

'''
D16. [EDA] Explore the Tokenized datasets
'''
print('질문 데이터의 크기(shape) :', padded_train_tkn.shape)

# 0번째 샘플을 임의로 출력
print(padded_train_tkn[0])

'''
D17. Split Data
'''
X_train = padded_train_tkn[:125000]
Y_train = Label_Train[:125000]
X_valid = padded_train_tkn[125000:]
Y_valid = Label_Train[125000:]
X_test  = padded_test_tkn
Y_test  = Label_test

print('Number of sequences for training dataset   : {}'.format(len(X_train)))
print('Number of sequences for validation dataset : {}'.format(len(X_valid)))
print('Number of sequences for testing dataset    : {}'.format(len(X_test)))

'''
D18. [PASS] Build dataset
'''
# For eager mode, it is done at the "model.fit"

'''
D19. [PASS] Define some useful parameters for further use
'''

'''
Model Engineering
'''

'''
M01. Import Libraries for Model Engineering
'''

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import load_model

from tensorflow import keras
from tensorflow.keras import layers

'''
M02. TPU Initialization
'''

import tensorflow as tf

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

'''
M03. Define Hyperparameters for Model Engineering
'''
embedding_dim = 256
hidden_size = 128
output_dim = 1  # output layer dimensionality = num_classes
EPOCHS = 20
batch_size = 100
learning_rate = 5e-4

'''
M04. Open "strategy.scope(  )"
'''

# initialize and compile model within strategy scope
with strategy.scope():
    '''
    M05. Build NN model
    '''
    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a embedding_dim-dimensional vector
    x = layers.Embedding(vocab_size, embedding_dim)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(hidden_size, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(hidden_size))(x)
    # Add a classifier
    outputs = layers.Dense(output_dim, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    
    '''
    M06. Optimizer
    '''
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    '''
    M07. Model Compilation - model.compile
    '''
    # model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer=optimizer, loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

model.summary()

'''
M08. EarlyStopping
'''
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 8)

'''
M09. ModelCheckpoint
'''
mc = ModelCheckpoint('best_model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)

'''
M10. Train and Validation - `model.fit`
'''
history = model.fit(X_train, Y_train, epochs = EPOCHS,
                    batch_size=batch_size,
                    validation_data = (X_valid, Y_valid),
                    verbose=1,
                    callbacks=[es, mc])

'''
M11. Assess model performance
'''
loaded_model = load_model('best_model.h5')
print("\n Test Accuracy: %.4f" % (loaded_model.evaluate(X_test, Y_test)[1]))

'''
M12. [Opt] Plot Loss and Accuracy
'''
history_dict = history.history
history_dict.keys()

acc      = history_dict['accuracy']
val_acc  = history_dict['val_accuracy']
loss     = history_dict['loss']
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

'''
M13. [Opt] Training result test for Code Engineering
'''
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
    if(Y_test[20000+idx] > 0.5):
        print("Ground truth is positive!")
    else:
        print("Ground truth is negative!")
    
    
