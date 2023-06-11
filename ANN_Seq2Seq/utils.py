# %%
import re
import spacy
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.datasets import Multi30k


symbols = set(['-', '(', '笑声', ')', '）', '（', '鼓掌'])

def load_dataset(batch_size):
    print("Batch_Size:", batch_size)
    # spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
    spacy_zh = spacy.load('zh_core_web_sm')
    # url = re.compile('(<url>.*</url>)')

    # def tokenize_de(text):
    #     return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]


    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text) if not tok.is_space and tok not in symbols]

    def tokenize_zh(text):
        return [tok.text for tok in spacy_zh.tokenizer(text) if not tok.is_space and tok not in symbols]

    ZH = Field(tokenize=tokenize_zh, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')


    train, val, test = TabularDataset.splits(
        path='./data/', train='train.csv',
        validation='dev.csv', test='test.csv', format='csv',
        fields=[('src', EN), ('trg', ZH)])

    data1 = train.examples[1]

    print(data1.src)
    print(data1.trg)


    # train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))
    EN.build_vocab(train.src, min_freq=2)
    ZH.build_vocab(train.trg, min_freq=2)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, repeat=False, sort=False)
    # train_iter, val_iter, test_iter = BucketIterator.splits(
    #     (train, val, test), batch_size=2, repeat=False, sort_within_batch=True, sort_key=lambda x: (len(x.src), len(x.trg)))
    return train_iter, val_iter, test_iter, EN, ZH

# %%
if __name__ == '__main__':
    train_iter, val_iter, test_iter, EN, ZH=load_dataset(32)
    print(len(ZH.vocab))