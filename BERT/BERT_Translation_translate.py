#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' BERT_translation '

__author__ = 'https://blog.51cto.com/u_11466419/5983209'

# python==3.7.16
# torch==1.11.0
# protobuf==3.20.1
# tokenizers==0.12.1
# torchtext==0.12.0
# tensorboard==2.8.0
# torch 和 torchtext 版本的对应关系：https://aitechtogether.com/ai-question/23173.html
# 导入环境
import os
import math
import numpy as np
import torch
# print(torch.__version__)
import torch.nn as nn
# hugging face的分词器，github地址：https://github.com/huggingface/tokenizers
from tokenizers import Tokenizer
# 用于构建词典
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import pad, log_softmax
from pathlib import Path
# from tqdm import tqdm
import tqdm

import nltk
import jieba
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

def en_tokenizer(line):
    """
    定义英文分词器，后续也要使用
    :param line: 一句英文句子，例如"I'm learning Deep learning."
    :return: subword分词后的记过，例如：['i', "'", 'm', 'learning', 'deep', 'learning', '.']
    """
    # 使用bert进行分词，并获取tokens。add_special_tokens是指不要在结果中增加‘<bos>’和`<eos>`等特殊字符
    return tokenizer.encode(line, add_special_tokens=False).tokens

# print(en_tokenizer("Are you crazy?"))

# 构建模型
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model).to(device)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
class TranslationModel(nn.Module):

    def __init__(self, d_model, src_vocab, tgt_vocab, dropout=0.1):
        super(TranslationModel, self).__init__()

        # 定义原句子的embedding
        self.src_embedding = nn.Embedding(len(src_vocab), d_model, padding_idx=2)
        # 定义目标句子的embedding
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, padding_idx=2)
        # 定义posintional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=max_length)
        # 定义Transformer
        self.transformer = nn.Transformer(d_model, dropout=dropout, batch_first=True)

        # 定义最后的预测层，这里并没有定义Softmax，而是把他放在了模型外。
        self.predictor = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt):
        """
        进行前向传递，输出为Decoder的输出。注意，这里并没有使用self.predictor进行预测，
        因为训练和推理行为不太一样，所以放在了模型外面。
        :param src: 原batch后的句子，例如[[0, 12, 34, .., 1, 2, 2, ...], ...]
        :param tgt: 目标batch后的句子，例如[[0, 74, 56, .., 1, 2, 2, ...], ...]
        :return: Transformer的输出，或者说是TransformerDecoder的输出。
        """

        """
        生成tgt_mask，即阶梯型的mask，例如：
        [[0., -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0.]]
        tgt.size()[-1]为目标句子的长度。
        """
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(device)
        # 掩盖住原句子中<pad>的部分，例如[[False,False,False,..., True,True,...], ...]
        src_key_padding_mask = TranslationModel.get_key_padding_mask(src)
        # 掩盖住目标句子中<pad>的部分
        tgt_key_padding_mask = TranslationModel.get_key_padding_mask(tgt)

        # 对src和tgt进行编码
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        return tokens == 2

# Transformer推理时，tgt是一次一个的将token传给Transformer，例如，首次tgt为​​<bos>​​​，然后预测出​​I​​​，然后第二次tgt为​​<bos> I​​​，预测出​​like​​​，第三次tgt为​​<bos> I like​​​，以此类推，直到预测结果为​​<eos>​​，或者达到句子最大长度。
def translate(src: str):
    """
    :param src: 英文句子，例如 "I like machine learning."
    :return: 翻译后的句子，例如：”我喜欢机器学习“
    """
    # print(src)
    # 将与原句子分词后，通过词典转为index，然后增加<bos>和<eos>
    src = torch.tensor([0] + en_vocab(en_tokenizer(src)) + [1]).unsqueeze(0).to(device)
    # 首次tgt为<bos>
    tgt = torch.tensor([[0]]).to(device)
    # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
    for i in range(max_length):
        # 进行transformer计算
        out = model(src, tgt)
        # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
        predict = model.predictor(out[:, -1])
        # 找出最大值的index
        y = torch.argmax(predict, dim=1)
        # 和之前的预测结果拼接到一起
        tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
        # 如果为<eos>，说明预测结束，跳出循环
        if y == 1:
            break
    # 将预测tokens拼起来
    tgt = ''.join(zh_vocab.lookup_tokens(tgt.squeeze().tolist())).replace("<s>", "").replace("</s>", "")
    return tgt

if __name__=="__main__":

    # 工作目录，缓存文件盒模型会放在该目录下
    work_dir = Path("./dataset")
    max_length = 128
    # 定义batch_size，由于是训练文本，占用内存较小，可以适当大一些
    batch_size = 64
    # epochs数量，不用太大，因为句子数量较多
    epochs = 10
    # 多少步保存一次模型，防止程序崩溃导致模型丢失。
    save_after_step = 200
    # 定义训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载基础的分词器模型，使用的是基础的bert模型。`uncased`意思是不区分大小写
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

    # 模型推理
    model_path = "./model/model_750000.pt"
    print(model_path)
    model = torch.load(model_path,map_location=torch.device('cpu'))
    model.eval()

    # 指定英文词典缓存文件路径
    # 如果使用缓存，且缓存文件存在，则加载缓存文件
    # 是否使用缓存，由于文件较大，初始化动作较慢，所以将初始化好的文件持久化
    use_cache = True
    en_vocab_file = work_dir / "vocab_en.pt"
    if use_cache and os.path.exists(en_vocab_file):
        en_vocab = torch.load(en_vocab_file, map_location="cpu")

    zh_vocab_file = work_dir / "vocab_zh.pt"
    if use_cache and os.path.exists(zh_vocab_file):
        zh_vocab = torch.load(zh_vocab_file, map_location="cpu")

    tgt = translate("I like machine learning.")
    print(tgt)
