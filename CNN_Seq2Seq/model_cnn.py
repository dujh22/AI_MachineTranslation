import torch.nn as nn
import torch.nn.functional as F
import torch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self,src_vocab_size,emb_size,hid_size,kernel_size,n_layers,dropout=0.25,max_len=100):

        super(Encoder,self).__init__()
        self.token_emb=nn.Embedding(src_vocab_size,emb_size)
        self.pos_emb=nn.Embedding(max_len,emb_size)
        
        self.emb2hid=nn.Linear(emb_size,hid_size)
        self.hid2emb=nn.Linear(hid_size,emb_size)
        
        self.convs=nn.ModuleList([
            nn.Conv1d(in_channels=hid_size,
                     out_channels=hid_size*2,
                     kernel_size=kernel_size,
                     padding=(kernel_size-1)//2)
            for _  in range(n_layers)
        ])
        self.dropout=nn.Dropout(dropout)
        self.scale=torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
    def forward(self, src):

        batch_size=src.shape[0]
        src_len=src.shape[1]
        
        pos=torch.arange(0,src_len).to(device)
        pos=pos.unsqueeze(0).repeat(batch_size,1)
        src_embed=self.token_emb(src)
        pos_embed=self.pos_emb(pos)

        src_pos_embed=self.dropout(src_embed+pos_embed)
        
        conv_input=self.emb2hid(src_pos_embed)
        conv_input=conv_input.permute(0,2,1)

        for conv in self.convs:
            conved=conv(self.dropout(conv_input))
            conved=F.glu(conved,dim=1)
            conved=(conved+conv_input)*self.scale
            conv_input=conved
        
        conved=conved.permute(0,2,1)
        conved=self.hid2emb(conved)
        combined=(conved+src_pos_embed)*self.scale

        return conved,combined

class Attention(nn.Module):
    def __init__(self,emb_size,hid_size):
        super(Attention,self).__init__()
        self.emb2hid=nn.Linear(emb_size,hid_size)
        self.hid2emb=nn.Linear(hid_size,emb_size)
        self.scale=torch.sqrt(torch.FloatTensor([0.5])).to(device)
    
    def forward(self,dec_conved,embedd,en_conved,en_combined):

        dec_conved=dec_conved.permute(0,2,1)
        dec_conved_emb=self.hid2emb(dec_conved)

        Q=(dec_conved_emb+embedd)*self.scale

        energy=torch.matmul(Q,en_conved.permute(0,2,1))

        a=F.softmax(energy,dim=2)

        context=torch.matmul(a,en_combined)
        context=self.emb2hid(context)
        conved=(context+dec_conved)*self.scale

        return conved.permute(0,2,1),a

class Decoder(nn.Module):
    def __init__(self,trg_vocab_size,emb_size,hid_size,kernel_size,n_layers,attnModel,dropout=0.25,max_len=50):

        super(Decoder,self).__init__()
        self.attnModel=attnModel
        self.kernel_size=kernel_size#要根据其在前面创建kernel-1个pad
        
        self.token_embed=nn.Embedding(trg_vocab_size,emb_size)
        self.pos_embed=nn.Embedding(max_len,emb_size)
        
        self.emb2hid=nn.Linear(emb_size,hid_size)
        self.hid2emb=nn.Linear(hid_size,emb_size)
        
        self.fc=nn.Linear(emb_size,trg_vocab_size)
        
        self.scale=torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.convs=nn.ModuleList([
            nn.Conv1d(in_channels=hid_size,
                     out_channels=2*hid_size,
                     kernel_size=kernel_size)
            for _ in range(n_layers)])
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,trg,en_conved,en_combined):
        
        batch_size=trg.shape[0]
        trg_len=trg.shape[1]
        
        pos=torch.arange(0,trg_len).to(device)
        pos=pos.unsqueeze(0).repeat(batch_size,1)

        token_embed=self.token_embed(trg)
        pos_embed=self.pos_embed(pos)
        
        embedd=self.dropout(token_embed+pos_embed)

        input_conv=self.emb2hid(embedd).permute(0,2,1)

        hid_size=input_conv.shape[1]
        for _,conv in enumerate(self.convs):
            input_conv=self.dropout(input_conv)
            padding=torch.ones(batch_size,hid_size,self.kernel_size-1).to(device)
            pad_input_conv=torch.cat((padding,input_conv),dim=2)
            
            conved=conv(pad_input_conv)
            conved=F.glu(conved,dim=1)
            conved,a=self.attnModel(conved,embedd,en_conved,en_combined)
            
            conved=(conved+input_conv)*self.scale

            input_conv=conved
        
        output=self.hid2emb(conved.permute(0,2,1))
        output=self.fc(self.dropout(output))
        return output,a


class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(Seq2Seq,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
    
    def forward(self,src,trg):
        en_coved,en_combined=self.encoder(src)
        output,attn=self.decoder(trg,en_coved,en_combined)
        return output,attn