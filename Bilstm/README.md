# Description

本项目是一个**基于神经网络的英-中机器翻译模型**，并制作了简洁的网页。
本项目全部自己自己手写实现，主要部分包括：LSTM,Bi-LSTM，Encoder,Decoder,Beam Search，全部从头实现。

# Get Started

- 环境配置
  - python=3.7.11
  - pytorch=1.0.0
  - torchvision
  - numpy
  - scipy
  - numpy
  - nltk
  - tqdm
  - pyinstaller
  
- 建立词典：`python run.py vocab`
- 训练： `python run.py train` (GPU)
- 测试：`python run.py test` (CPU)


# Dataset

本实验使用的数据集为`IWSLT 2017`TED数据集，包含240264条中英平行语料，训练：验证：测试集比例为 0.6: 0.2: 0.2.


# NMT_Model

## 模型架构

本实验使用经典的`Seq2seq`网络，即`Encoder-Decoder`架构。`Encoder`采用双向`LSTM`，`Decoder`采用单向`LSTM`。

普通的`seq2seq`模型`Decoder`过分依赖于`Encoder`在最后`time step`生成的向量（`context vector`），而实际上，`Decoder`在生成输出序列中的每一个词时，可能只需利用输入序列的某一部分信息。由此，我们引入了`Attention`机制，以实现让模型能够根据信息的有用程度分配权重，更好处理长距离依赖问题。


## 训练

### 参数设置

训练过程的参数设置如下：

| batch_size    | beam_size      | clip_grad       | dropout      | lr                | lr_decay      |
| ------------- | -------------- | --------------- | ------------ | ----------------- | ------------- |
| 32            | 5              | 5.0             | 0.3          | 0.001             | 0.5           |
| **max_epoch** | **embed_size** | **hidden_size** | **patience** | **num_max_trial** | **Optimizer** |
| 30            | 256            | 256             | 5            | 5                 | Adam          |


### 训练技术

#### Beam Search

传统的贪心做法时，每次选择输出概率最大的一个单词，但局部最优解无法保证全局最优解。而在采用`Beam search`的情况下，每次会选择`beam_size`个概率最大的单词（本实验中`beam_size ` = 5），然后进行下一步，依次类推。

#### Gradient Clipping

在模型训练过程中，梯度可能变得太大而使得模型参数不稳定，称为梯度爆炸问题。
为了解决梯度爆炸问题，采用`Gradient Clipping`技术。如果梯度变得非常大，我们就调节它使其保持较小的状态。
Gradient Clipping 确保了梯度矢量的最大范数。有助于梯度下降保持合理。

#### Early Stopping

采用`early stopping`方法，设置`patience`为5，`max_num_trial`为5，即：若模型在验证集上的测评效果连续变差次数超过5次，则重启到之前最好的模型参数和优化器参数，并减小学习率（本实验中`lr_decay` = 0.5），最多重启5次。 


## 模型测评

本实验使用`BLEU`对模型进行测评。`BLEU`是`NMT`系统最常用的自动测评矩阵，它通常基于整个测试集进行计算。


# 模型部署

我们写了一个网页，运行service.py启动本地服务器，在浏览器打开网页http://127.0.0.1:5000/即可进入翻译界面。




