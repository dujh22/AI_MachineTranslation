# %%
import torch
import torch.nn as nn
from model import Encoder, Decoder, Seq2Seq
from utils import load_dataset

train_iter, val_iter, test_iter, EN, ZH = load_dataset(32)
en_size, zh_size = len(EN.vocab), len(ZH.vocab)


# %%
hidden_size = 512
embed_size = 256

encoder = Encoder(en_size, embed_size, hidden_size,
                    n_layers=2, dropout=0.5)
decoder = Decoder(embed_size, hidden_size, zh_size,
                    n_layers=1, dropout=0.5)

model = Seq2Seq(encoder, decoder).cuda()

# %% 加载模型
model_path = '/nfs/home/zhaoyiyang/class/seq2seq/seq2seq-master/save_batchsize2046/seq2seq_5.pt'
model.load_state_dict(torch.load(model_path))

# %%
from tqdm import tqdm
vocab_size = zh_size
belu_score = []
with torch.no_grad():
    model.eval()
    pad = ZH.vocab.stoi['<pad>']
    total_loss = 0
    for b, batch in tqdm(enumerate(test_iter)):
        src, len_src = batch.src
        trg, len_trg = batch.trg

        for j in range(len(test_trg)):
            src_list = []
            tar_list = []
            out_list = []
            
            test_src = src.transpose(0, 1)
            test_trg = trg.transpose(0, 1)
            # for i in range(len(test_src[index])):
            #     print(EN.vocab.itos[test_src[index][i]],' ',end='')
            # print()
            for i in range(len(test_trg[j])):
                if ZH.vocab.itos[test_trg[j][i]] != '<pad>':
                    if ZH.vocab.itos[test_trg[j][i]] != '<sos>' and ZH.vocab.itos[test_trg[j][i]] != '<eos>':
                        tar_list.append(ZH.vocab.itos[test_trg[j][i]])
                # print(ZH.vocab.itos[test_trg[index][i]],' ',end='')
            # print()
            src = src.data.cuda()
            trg = trg.data.cuda()
            output = model(src, trg, teacher_forcing_ratio=0.0)

            # print(output.shape)
            # print(output[1:].view(-1, vocab_size).shape)


            indices = torch.argmax(output, dim=-1).transpose(0, 1)
            # max_indices = torch.unsqueeze(indices, dim=-1)

            answ = indices.cpu().numpy()
            for i in range(len(answ[j])):
                if ZH.vocab.itos[answ[j][i]] != '<pad>':
                    if ZH.vocab.itos[answ[j][i]] != '<sos>' and ZH.vocab.itos[answ[j][i]] != '<eos>':
                        out_list.append(ZH.vocab.itos[answ[j][i]])
                
            score1 = sentence_bleu(tar_list, out_list, weights=(1, 0, 0, 0))
            # score1 = sentence_bleu(tar_list, out_list, weights=(0, 1, 0, 0))
            # score3 = sentence_bleu(tar_list, out_list, weights=(0, 0, 1, 0))
            # score4 = sentence_bleu(tar_list, out_list, weights=(0, 0, 0, 1))
            belu_score.append(score1)
            # print('Cumulate 1-gram :%f' \
                # % score1)
            # print('Cumulate 2-gram :%f' \
            #     % score2)
            # print('Cumulate 3-gram :%f' \
            #     % score3)
            # print('Cumulate 4-gram :%f' \
            #     % score4)
        print(sum(belu_score)/len(belu_score) * 100)
print(sum(belu_score)/len(belu_score) * 100)


                # print(ZH.vocab.itos[answ[index][i]],' ',end='')

# %%
from nltk.translate.bleu_score import sentence_bleu
model.eval()
belu_score = []
with torch.no_grad():
    for example in test_iter:
        src=example.src
        trg=example.trg

        output,_=model(src,trg[:,:-1])

        index = 2
        src_list = []
        tar_list = []
        out_list = []

        for i in range(len(src[index])):
            src_list.append(SRC.vocab.itos[src[index][i]])

        for i in range(len(trg[index])):
            if TRG.vocab.itos[trg[index][i]] != '<pad>':
                if TRG.vocab.itos[trg[index][i]] != '<sos>' and TRG.vocab.itos[trg[index][i]] != '<eos>':
                    tar_list.append(TRG.vocab.itos[trg[index][i]])
            else:
                break


        outputlist = torch.argmax(output, dim=-1)
        answ = outputlist.cpu().numpy()
        for i in range(len(answ[index])):
            if TRG.vocab.itos[answ[index][i]] != '<pad>':
                if TRG.vocab.itos[answ[index][i]] != '<sos>' and TRG.vocab.itos[answ[index][i]] != '<eos>':
                    out_list.append(TRG.vocab.itos[answ[index][i]])
            else:
                break


        score1 = sentence_bleu(tar_list, out_list, weights=(1, 0, 0, 0))
        # score1 = sentence_bleu(tar_list, out_list, weights=(0, 1, 0, 0))
        # score3 = sentence_bleu(tar_list, out_list, weights=(0, 0, 1, 0))
        # score4 = sentence_bleu(tar_list, out_list, weights=(0, 0, 0, 1))
        belu_score.append(score1)
        print('Cumulate 1-gram :%f' \
            % score1)
        # print('Cumulate 2-gram :%f' \
        #     % score2)
        # print('Cumulate 3-gram :%f' \
        #     % score3)
        # print('Cumulate 4-gram :%f' \
        #     % score4)
print(sum(belu_score)/len(belu_score) * 100)

# %%
vocab_size = zh_size
with torch.no_grad():
    model.eval()
    pad = ZH.vocab.stoi['<pad>']
    total_loss = 0
    for b, batch in enumerate(test_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg

        
        test_src = src.transpose(0, 1)
        test_trg = trg.transpose(0, 1)
        index = 11
        # for i in range(len(test_src[index])):
        #     print(EN.vocab.itos[test_src[index][i]],' ',end='')
        # print()
        for i in range(len(test_trg[index])):
            print(ZH.vocab.itos[test_trg[index][i]],' ',end='')
        print()
        src = src.data.cuda()
        trg = trg.data.cuda()
        output = model(src, trg, teacher_forcing_ratio=0.0)

        print(output.shape)
        print(output[1:].view(-1, vocab_size).shape)


        indices = torch.argmax(output, dim=-1).transpose(0, 1)
        # max_indices = torch.unsqueeze(indices, dim=-1)

        answ = indices.cpu().numpy()
        for i in range(len(answ[index])):
            print(ZH.vocab.itos[answ[index][i]],' ',end='')

        break