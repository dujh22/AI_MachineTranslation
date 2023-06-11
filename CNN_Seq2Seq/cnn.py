# %%
import torch
import torch.nn as nn
import spacy
from torchtext.data import Field, BucketIterator, TabularDataset
from model_cnn import Encoder,Attention,Decoder,Seq2Seq

# %%
en_seq=spacy.load("en_core_web_sm")
zh_seq=spacy.load("zh_core_web_sm")
 
symbols = set(['-', '(', '笑声', ')', '）', '（', '鼓掌', ',', '.', ';', '，', '。', '?', '？'])

def src_token(text):
    return [word.text for word in en_seq.tokenizer(text) if not word.is_space and word not in symbols]
 
def trg_token(text):
    return [word.text for word in zh_seq.tokenizer(text) if not word.is_space and word not in symbols]
 
SRC=Field(tokenize=src_token,
         init_token="<sos>",
         eos_token="<eos>",
         lower=True,
         batch_first=True,
         fix_length=26)
 
TRG=Field(tokenize=trg_token,
         init_token="<sos>",
         eos_token="<eos>",
         lower=True,
         batch_first=True,
         fix_length=26)

train_data, val_data, test_data = TabularDataset.splits(
        path='./data/', train='train.csv',
        validation='dev.csv', test='test.csv', format='csv',
        fields=[('src', SRC), ('trg', TRG)])

# %%
SRC.build_vocab(train_data,min_freq=2)
TRG.build_vocab(train_data,min_freq=2)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")
print(device)
batch_size=128

train_iter,val_iter,test_iter=BucketIterator.splits(
    (train_data,val_data,test_data),
    batch_size=batch_size,
    device=device,
    sort=False
)

# %%

import torch.nn as nn
import torch.nn.functional as F


src_vocab_size=len(SRC.vocab)
trg_vocab_size=len(TRG.vocab)
 
emb_size=256
hid_size=512
kernel_size=3
n_layers=10

# %%
enModel=Encoder(src_vocab_size,emb_size,hid_size,kernel_size,n_layers).to(device)
attModel=Attention(emb_size,hid_size).to(device)
deModel=Decoder(trg_vocab_size,emb_size,hid_size,kernel_size,n_layers,attModel).to(device)
model=Seq2Seq(enModel,deModel).to(device)


import math,time
from torch.optim import Adam

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

epochs=30
clip=0.1
criterion=nn.CrossEntropyLoss(ignore_index=1)
optim=Adam(model.parameters())

# %%
def train(model,data_iter,criterion,optim,clip):
    
    model.train()
    lossAll=0
    temp = 0
    lossTemp = 0
    for example in data_iter:
        temp += 1
        src=example.src
        trg=example.trg
        
        optim.zero_grad()
        output,_=model(src,trg[:,:-1])
        #output[batch trg_len-1 trg_vocab_size]
        output=output.reshape(-1,trg_vocab_size)
        trg=trg[:,1:].reshape(-1)
        #output[batch*(trg_len-1),trg_vocab_size]
        #trg[batch*(trg_ken-1)]
        loss=criterion(output,trg)
        loss.backward()      
        torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        optim.step()
        
        lossAll+=loss.item()
        lossTemp+=loss.item()
        if temp % 100 == 0 and temp != 0:
            lossTemp = lossTemp/100
            print("[%d][loss:%5.2f][pp:%5.2f]" % (temp, lossTemp, math.exp(lossTemp)))
            file.write("[%d][loss:%5.2f][pp:%5.2f]" % (temp, lossTemp, math.exp(lossTemp)))
            lossTemp = 0
    return lossAll/len(data_iter)


def evaluate(model,data_iter,criterion):
    
    model.eval()
    lossAll=0
    
    with torch.no_grad():
        for example in data_iter:
            src=example.src
            trg=example.trg
 
            output,_=model(src,trg[:,:-1])
            output=output.reshape(-1,trg_vocab_size)
            trg=trg[:,1:].reshape(-1)
            loss=criterion(output,trg)
            lossAll+=loss.item()
    return lossAll/len(data_iter)

print('start')
file = open("./cnn_chickpoint/loss.txt","w")
for epoch in range(epochs):
    
    start_time = time.time()
    train_loss = train(model,train_iter,criterion,optim,clip)
    valid_loss = evaluate(model,val_iter,criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    file.write(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    file.write(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    file.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    torch.save(model.state_dict(), './cnn_chickpoint/CNN_%d.pt' % (epoch))