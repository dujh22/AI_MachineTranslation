class HMMTranslator:
    def __init__(self):
        self.trans_probs = {}
        self.emit_probs = {}

    def train(self, source_sentences, target_sentences):
        # 计算转移概率
        for source_sent, target_sent in zip(source_sentences, target_sentences):
            source_words = source_sent.split()
            target_words = target_sent.split()
            for i in range(len(source_words)):
                source_word = source_words[i]
                target_word = target_words[i]
                if source_word not in self.trans_probs:
                    self.trans_probs[source_word] = {}
                if target_word not in self.trans_probs[source_word]:
                    self.trans_probs[source_word][target_word] = 0
                self.trans_probs[source_word][target_word] += 1

        # 归一化转移概率
        for source_word in self.trans_probs:
            total_count = sum(self.trans_probs[source_word].values())
            for target_word in self.trans_probs[source_word]:
                self.trans_probs[source_word][target_word] /= total_count

        # 计算发射概率
        for source_sent, target_sent in zip(source_sentences, target_sentences):
            source_words = source_sent.split()
            target_words = target_sent.split()
            for i in range(min(len(source_words), len(target_words))):
                source_word = source_words[i]
                target_word = target_words[i]
                if source_word not in self.emit_probs:
                    self.emit_probs[source_word] = {}
                if target_word not in self.emit_probs[source_word]:
                    self.emit_probs[source_word][target_word] = 0
                self.emit_probs[source_word][target_word] += 1

        # 归一化发射概率
        for source_word in self.emit_probs:
            total_count = sum(self.emit_probs[source_word].values())
            for target_word in self.emit_probs[source_word]:
                self.emit_probs[source_word][target_word] /= total_count

    def translate(self, source_sentence):
        source_words = source_sentence.split()
        target_words = []
        for source_word in source_words:
            if source_word in self.trans_probs:
                target_word = max(self.trans_probs[source_word], key=self.trans_probs[source_word].get)
                target_words.append(target_word)
            else:
                target_words.append(source_word)
        return " ".join(target_words)


# 创建并训练模型
translator = HMMTranslator()
translator.train(chinese_sentences, english_sentences)

# 使用模型进行翻译
source_sentence = "去年我给各位展示了两个"
translation = translator.translate(source_sentence)
print(f"Source: {source_sentence}")
print(f"Translation: {translation}")

#%%
class NGramHMMTranslator:
    def __init__(self, n=2, smoothing=1.0):
        self.n = n  # n-gram size
        self.smoothing = smoothing  # smoothing parameter
        self.trans_probs = {}
        self.emit_probs = {}

    def train(self, source_sentences, target_sentences):
        # compute transition probabilities
        for source_sent, target_sent in zip(source_sentences, target_sentences):
            source_ngrams = [' '.join(source_sent.split()[i:i+self.n]) for i in range(len(source_sent.split())-self.n+1)]
            target_ngrams = [' '.join(target_sent.split()[i:i+self.n]) for i in range(len(target_sent.split())-self.n+1)]
            for source_ngram, target_ngram in zip(source_ngrams, target_ngrams):
                if source_ngram not in self.trans_probs:
                    self.trans_probs[source_ngram] = {}
                if target_ngram not in self.trans_probs[source_ngram]:
                    self.trans_probs[source_ngram][target_ngram] = 0
                self.trans_probs[source_ngram][target_ngram] += 1

        # normalize transition probabilities
        for source_ngram in self.trans_probs:
            total_count = sum(self.trans_probs[source_ngram].values())
            for target_ngram in self.trans_probs[source_ngram]:
                self.trans_probs[source_ngram][target_ngram] = (self.trans_probs[source_ngram][target_ngram] + self.smoothing) / (total_count + self.smoothing * len(self.trans_probs[source_ngram]))

    def translate(self, source_sentence):
        source_ngrams = [' '.join(source_sentence.split()[i:i+self.n]) for i in range(len(source_sentence.split())-self.n+1)]
        target_sentence = []
        for source_ngram in source_ngrams:
            if source_ngram in self.trans_probs:
                target_ngram = max(self.trans_probs[source_ngram], key=self.trans_probs[source_ngram].get)
                target_sentence.append(target_ngram)
            else:
                target_sentence.append(source_ngram)  # use source ngram if no translation found
        return ' '.join(target_sentence)


# create and train the model
translator = NGramHMMTranslator()
translator.train(chinese_sentences, english_sentences)

#%%
source_sentence = "非常谢谢克里斯的确非常荣幸 能有第二次站在这个台上的机会我真是非常感激"
translation = translator.translate(source_sentence)
print(f"Source: {source_sentence}")
print(f"Translation: {translation}")