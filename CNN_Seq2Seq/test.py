# %%
import torch
import torch.nn as nn
import spacy
from torchtext.data import Field, BucketIterator, TabularDataset
from model_cnn import Encoder,Attention,Decoder,Seq2Seq
from nltk.translate.bleu_score import sentence_bleu

batch_size=128
emb_size=256
hid_size=512
kernel_size=3
n_layers=10

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


train_iter,val_iter,test_iter=BucketIterator.splits(
    (train_data,val_data,test_data),
    batch_size=batch_size,
    device=device,
    sort=False
)

import torch.nn as nn

src_vocab_size=len(SRC.vocab)
trg_vocab_size=len(TRG.vocab)
 
# %%

enModel=Encoder(src_vocab_size,emb_size,hid_size,kernel_size,n_layers).to(device)
attModel=Attention(emb_size,hid_size).to(device)
deModel=Decoder(trg_vocab_size,emb_size,hid_size,kernel_size,n_layers,attModel).to(device)
model=Seq2Seq(enModel,deModel).to(device)

# %% 加载模型
model_path = '/nfs/home/zhaoyiyang/class/temp/cnn_chickpoint/CNN_8.pt'
model.load_state_dict(torch.load(model_path))

# %% PPL
import math
criterion=nn.CrossEntropyLoss(ignore_index=1)
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

loss_test = evaluate(model, test_iter, criterion)
print(math.exp(loss_test))
# %% 

criterion=nn.CrossEntropyLoss(ignore_index=1)

def evaluate(model,data_iter,criterion):
    model.eval()
    lossAll=0
    
    with torch.no_grad():
        for example in data_iter:
            src=example.src
            trg=example.trg
 
            output,_=model(src,trg[:,:-1])

            index = 28
            for i in range(len(src[index])):
                print(SRC.vocab.itos[src[index][i]],' ',end='')
            print()
            print()
            for i in range(len(trg[index])):
                print(TRG.vocab.itos[trg[index][i]],' ',end='')
            print()
            print()
 
            print(output.shape)

            outputlist = torch.argmax(output, dim=-1)
            answ = outputlist.cpu().numpy()
            for i in range(len(answ[index])):
                print(TRG.vocab.itos[answ[index][i]],' ',end='')

            break
    return lossAll/len(data_iter)

evaluate(model,test_iter,criterion)
print('\n---------------------------\n')
evaluate(model,train_iter,criterion)

# %% BELU
from nltk.translate.bleu_score import sentence_bleu
model.eval()
belu_score = []
with torch.no_grad():
    for example in test_iter:
        src=example.src
        trg=example.trg

        output,_=model(src,trg[:,:-1])

        for j in range(len(src)):
            src_list = []
            tar_list = []
            out_list = []

            for i in range(len(src[j])):
                src_list.append(SRC.vocab.itos[src[j][i]])

            for i in range(len(trg[j])):
                if TRG.vocab.itos[trg[j][i]] != '<pad>':
                    if TRG.vocab.itos[trg[j][i]] != '<sos>' and TRG.vocab.itos[trg[j][i]] != '<eos>':
                        tar_list.append(TRG.vocab.itos[trg[j][i]])
                else:
                    break


            outputlist = torch.argmax(output, dim=-1)
            answ = outputlist.cpu().numpy()
            for i in range(len(answ[j])):
                if TRG.vocab.itos[answ[j][i]] != '<pad>':
                    if TRG.vocab.itos[answ[j][i]] != '<sos>' and TRG.vocab.itos[answ[j][i]] != '<eos>':
                        out_list.append(TRG.vocab.itos[answ[j][i]])
                else:
                    break


            score = sentence_bleu(tar_list, out_list, weights=(1, 0, 0, 0))
            belu_score.append(score)

        print(sum(belu_score)/len(belu_score) * 100)
print(sum(belu_score)/len(belu_score) * 100)