!pip install -U torchtext==0.10.0

! git clone https://github.com/simonjisu/nsmc_study.git
    
import torch
from torchtext.legacy import data
import torch.nn.functional as F

import random
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split    

import pandas as pd
import torch.optim as optim
import re

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
%cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab190912.sh

from konlpy.tag import Mecab

tokenizer = Mecab()

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

train_data, valid_data = train_test_split(train_data, test_size=0.3, random_state=32)

print(len(train_data))
print(len(valid_data))
print(len(test_data))
print(train_data.shape)
print(valid_data.shape)
print(test_data.shape)

TEXT = data.Field(sequential=True, use_vocab=True, tokenize=tokenizer.morphs, lower=False, batch_first=True, fix_length=20)
LABEL = data.LabelField(dtype = torch.float)

def convert_dataset(input_data, text, label):
    list_of_example = [data.Example.fromlist(row.tolist(), fields=[('text', text), ('label', label)])  for _, row in input_data.iterrows()]
    dataset = data.Dataset(examples=list_of_example, fields=[('text', text), ('label', label)])
    return dataset

train_data = convert_dataset(train_data,TEXT,LABEL)
valid_data = convert_dataset(valid_data, TEXT, LABEL)
test_data = convert_dataset(test_data, TEXT, LABEL)

print(f'Number of training examples   : {len(train_data)}')
print(f'Number of validation examples : {len(valid_data)}')
print(f'Number of testing examples    : {len(test_data)}')

MAX_VOCAB_SIZE = 20000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)

LABEL.build_vocab(train_data)

print(f"Unique tokens in TEXT vocabulary : {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

print(TEXT.vocab.freqs.most_common(20))

print(TEXT.vocab.itos[:10])

print(LABEL.vocab.stoi)

BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = data.Iterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    sort = False,
    device = device)

print('Number of minibatch for training dataset   : {}'.format(len(train_iterator)))
print('Number of minibatch for validation dataset : {}'.format(len(valid_iterator)))
print('Number of minibatch for testing dataset    : {}'.format(len(test_iterator)))

class CNN_network(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                   n_filters, filter_sizes,
                   output_dim,
                   dropout,
                   pad_idx          #<pad> token
                   ):

        super(CNN_network, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        # set n_filters == embeding_dim
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=filter_sizes, padding='same')
        self.dropout = nn.Dropout(dropout)

        self.attention = Self_attn(embedding_dim, n_filters, dropout)
        self.layernorm = nn.LayerNorm(n_filters)

        self.fc = nn.Linear(n_filters, output_dim)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        #embedded : batch x seq len x embed dim

        embedded = embedded.permute(0,2,1).contiguous()
        #embedded : batch x embed dim x seq len

        conv_out = F.relu(self.conv(embedded))
        #conv_out : batch x feature map x (sent len - filter_size[n] + 1)
        # == batch * feature map * seq len

        conv_out = conv_out.permute(2,0,1)
        # conv_out = seq len x batch x feature map

        attn, _ =  self.attention(conv_out)
        # attn = seq len x batch x channel
        output = self.layernorm(attn) + conv_out
        # output = seq len x batch x channel

        output = output.permute(1, 2, 0).contiguous()
        # output = batch x channel x seq len

        pooled = F.max_pool1d(output, output.shape[2]).squeeze(2)
        # pooled = batch x channel

        output = self.fc(pooled)

        return output

class Self_attn(nn.Module):
    def __init__(self, emb_dim, out_channels, dropout):
        super(Self_attn, self).__init__()

        self.embed_dim = emb_dim
        self.out_channels = out_channels

        self.in_proj_q = nn.Linear(self.out_channels, self.embed_dim, bias=True)
        self.in_proj_k = nn.Linear(self.out_channels, self.embed_dim, bias=True)
        self.in_proj_v = nn.Linear(self.out_channels, self.embed_dim, bias=True)

        self.out_proj = nn.Linear(self.embed_dim, self.out_channels, bias=True)

        self.dropout = nn.Dropout(p=dropout)
        self.scaling = self.embed_dim ** -0.5
    
    def forward(self, x):
        '''
        inputs :
          x : input feature map (seq len x batch x feature map)
        '''
        query = self.in_proj_q(x)
        key = self.in_proj_k(x)
        value = self.in_proj_v(x)
        # q,k,v = seq len x batch x channel(channel == embed_dim)

        query *= self.scaling

        query = query.transpose(0,1)
        key = key.transpose(0,1)
        value = value.transpose(0,1)

        attn_weight = torch.bmm(query, key.transpose(1,2))
        # attn_weight = batch x seq len x seq len

        attn_weight = F.softmax(attn_weight, dim=-1)
        # attn_weight = batch x seq len x seq len

        attn_weight = self.dropout(attn_weight)
        # attn_weight = batch x seq len x seq len

        attn = torch.bmm(attn_weight, value)
        # attn = batch x seq len x channel

        attn = attn.transpose(0,1).contiguous()
        # attn = seq len x batch x channel

        attn = self.out_proj(attn)
        # attn = seq len x batch x channel

        return attn, attn_weight

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100

OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN_network(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, 3, OUTPUT_DIM,
            DROPOUT, PAD_IDX)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, target):
    '''
    from https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb
    '''
    # round predictions to the closest integer (0 or 1)
    rounded_preds = torch.round(torch.sigmoid(preds))

    #convert into float for division
    correct = (rounded_preds == target).float()

    # rounded_preds = [ 1   0   0   1   1   1   0   1   1   1]
    # targets       = [ 1   0   1   1   1   1   0   1   1   0]
    # correct       = [1.0 1.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0]
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        # We initialize the gradient to 0 for every batch.
        optimizer.zero_grad()

        # batch of sentences인 batch.text를 model에 입력
        predictions = model(batch.text).squeeze(1)
        
        # prediction결과와 batch.label을 비교하여 loss값 계산 
        loss = criterion(predictions, batch.label)

        # Accuracy calculation
        acc = binary_accuracy(predictions, batch.label)

        # backward()를 사용하여 역전파 수행
        loss.backward()

        # 최적화 알고리즘을 사용하여 parameter를 update
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    # "evaluation mode" : turn off "dropout" or "batch nomalizaation"
    model.eval()

    # Use less memory and speed up computation by preventing gradients from being computed in pytorch
    with torch.no_grad():
    
        for batch in iterator:
            
            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 10

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut4-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

model.load_state_dict(torch.load('tut4-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

import torch
model.load_state_dict(torch.load('tut4-model.pt'))

def predict(model,sentence):
    model.eval()
    with torch.no_grad():
        sent = tokenizer.morphs(sentence)
        sent = torch.tensor([TEXT.vocab.stoi[i] for i in sent])
        sent = F.pad(sent,pad = (1,50-len(sent)-1),value = 1)
        sent = sent.unsqueeze(dim = 0) #for batch
        output = torch.sigmoid(model(sent))

        return output.item()
        
examples = [
  "재미 있어요! 꼭 보세요!",
  "추천하기 어렵네요. 그다지 재미는 없는듯!",
  "너무 신나요. 인생 역작입니다. 강추"
]

model = model.to('cpu')
for idx in range(len(examples)) :

    sentence = examples[idx]
    pred = predict(model,sentence)
    print("\n",sentence)
    if pred >= 0.5 :
        print(f">>>긍정 리뷰입니다. ({pred : .2f})")
    else:
        print(f">>>부정 리뷰입니다.({pred : .2f})")

