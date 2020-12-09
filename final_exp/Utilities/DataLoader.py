import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.utils import shuffle
import numpy as np
from gensim.models import FastText

# 定义标签和数字的映射关系
label_to_number_a = {'OFF': 0, 'NOT': 1}
label_to_number_b = {'UNT': 0, 'TIN': 1}
label_to_number_c = {'IND': 0, 'OTH': 1, 'GRP': 2}
number_to_label_a = {v: k for k, v in label_to_number_a.items()}
number_to_label_b = {v: k for k, v in label_to_number_b.items()}
number_to_label_c = {v: k for k, v in label_to_number_c.items()}


class Data:
    """
    定义样本的数据结构
    """

    def __init__(self, x=None, labels=None):
        """
        初始化
        :param x: 文本向量
        :param labels: 数字标签
        """
        self.x = x
        self.labels = labels


def clean_data(tweet):
    split_tweet = tweet.lower().split()
    clean_tweet = []
    previous_word = None
    for word in split_tweet:
        word = re.sub("[#@]", "", word)
        word = re.sub("!", " !", word)
        word = re.sub("[?]", " ?", word)

        if word == "user" and previous_word == "user":
            pass
        else:
            clean_tweet.append(word)

        previous_word = word

    return " ".join(clean_tweet)


def balence_data(x_train, y_train, ratio_down_over_up=0.5):
    x_train = list(x_train)
    y_train = list(y_train)

    n_cat = len(Counter(y_train))

    sorted_counter = Counter(y_train).most_common()
    max_cat = sorted_counter[0][1]
    min_cat = sorted_counter[-1][1]

    target = min_cat + (1 - ratio_down_over_up) * (max_cat - min_cat)

    for i in range(n_cat):
        diff = int(sorted_counter[i][1] - target)
        k = 0
        if diff > 0:
            rm = 0
            while rm <= diff:
                if y_train[k] == sorted_counter[i][0]:
                    x_train.pop(k)
                    y_train.pop(k)
                    rm += 1
                    k -= 1
                k += 1
        else:
            ad = 0
            while ad <= -diff:
                if y_train[k] == sorted_counter[i][0]:
                    x_train.append(x_train[k])
                    y_train.append(y_train[k])
                    ad += 1
                k += 1

    return shuffle(np.array(x_train), np.array(y_train))


class DataLoader:
    """
    加载数据集并对进行预处理
    """

    def __init__(self, path, embedding_size=100):
        """
        初始化
        :param path: 数据集的存储目录
        """
        self.path = path

        # 3个task的训练集
        self.training_a = Data()
        self.training_b = Data()
        self.training_c = Data()
        # 3个task的验证集
        self.validation_a = Data()
        self.validation_b = Data()
        self.validation_c = Data()
        # 3个task的测试集
        self.test_a = Data()
        self.test_b = Data()
        self.test_c = Data()
        # 最大向量长度
        self.max_seq_len = int()
        # Embedding矩阵
        self.embedding_matrix = np.array([])
        # Embedding向量长度
        self.embedding_size = embedding_size
        # 词袋大小
        self.vocabulary_size = 0

    def activate(self):
        """
        计算并处理所需的全部数据
        :return:
        """
        self._preprocess()
        self._embed()
        self._load_test_set()
        print("Done!")

    def _preprocess(self):
        """
        对训练集进行预处理
        :return:
        """
        # 读取原始训练数据
        print("Loading raw training data")
        self._training_data = pd.read_csv(self.path + 'olid-training-v1.0.tsv', sep='\t', header=0)
        # 清洗数据，统一小写，标点符号前后加空格
        print("Cleaning training data")
        self._training_data = self._training_data.merge(
            self._training_data.tweet.apply(
                lambda x: pd.Series({'clean': clean_data(x)})),
            left_index=True, right_index=True)

        # 初始化文本向量化工具
        self._tokenizer = Tokenizer()
        # 文本向量化
        print("Tokenizing")
        self._tokenizer.fit_on_texts(self._training_data.clean)
        sequences = self._tokenizer.texts_to_sequences(self._training_data.clean)
        seq = pad_sequences(sequences)
        self.max_seq_len = len(seq[0])

        # 提取每个task的训练样本
        data_a = self._training_data[self._training_data["subtask_a"].notna()]
        data_b = self._training_data[self._training_data["subtask_b"].notna()]
        data_c = self._training_data[self._training_data["subtask_c"].notna()]

        # 分离每种样本的文本
        x_a = data_a[["clean"]]
        # label_a = data_a["subtask_a"]
        x_b = data_b[["clean"]]
        # label_b = data_b["subtask_b"]
        x_c = data_c[["clean"]]
        # label_c = data_c["subtask_c"]

        # 将字符标签对应到数字，便于训练
        binary_labels_a = np.array(data_a.subtask_a.apply(lambda x: label_to_number_a[x]))
        binary_labels_b = np.array(data_b.subtask_b.apply(lambda x: label_to_number_b[x]))
        binary_labels_c = np.array(data_c.subtask_c.apply(lambda x: label_to_number_c[x]))

        # 划分训练集和验证集
        print("Splitting sets")
        self.training_a.x, self.validation_a.x, self.training_a.labels, self.validation_a.labels = train_test_split(
            self._build_seq(x_a),
            binary_labels_a,
            test_size=0.2)
        self.training_b.x, self.validation_b.x, self.training_b.labels, self.validation_b.labels = train_test_split(
            self._build_seq(x_b),
            binary_labels_b,
            test_size=0.2)
        self.training_c.x, self.validation_c.x, self.training_c.labels, self.validation_c.labels = train_test_split(
            self._build_seq(x_c),
            binary_labels_c,
            test_size=0.2)

        # 对训练集进行上采样或下采样，均衡化每种标签的训练集大小
        print("Balancing training data")
        self.training_a.x, self.training_a.labels = balence_data(self.training_a.x, self.training_a.labels, 0.3)
        self.training_b.x, self.training_b.labels = balence_data(self.training_b.x, self.training_b.labels, 0.2)
        self.training_c.x, self.training_c.labels = balence_data(self.training_c.x, self.training_c.labels, 0.7)

    def _embed(self):
        """
        实现词向量的embedding
        :return:
        """
        print("Embedding")
        sequences = self._tokenizer.texts_to_sequences(self._training_data.clean)
        seq = pad_sequences(sequences)
        preprocessed = self._tokenizer.sequences_to_texts(seq)
        sentences = [t.split() for t in preprocessed]
        model = FastText(sentences, size=self.embedding_size, window=5, min_count=1, workers=4, iter=40)
        self.vocabulary_size = len(self._tokenizer.word_index)
        self.embedding_matrix = np.zeros((self.vocabulary_size + 1, self.embedding_size))
        for word, i in self._tokenizer.word_index.items():
            try:
                embedding_vector = model.wv.word_vec(word)
                if embedding_vector is not None:
                    self.embedding_matrix[i] = embedding_vector
            except KeyError:
                print(word)

    def _build_seq(self, data_set):
        """
        Text -> token
        :param data_set:
        :return:
        """
        sequences = self._tokenizer.texts_to_sequences(data_set.clean)
        seq = pad_sequences(sequences, maxlen=self.max_seq_len)
        return seq

    def _load_test_set(self):
        """
        加载测试集并进行预处理
        :return:
        """
        # 读入原始训练数据
        print("Loading test data")
        test_data_a = pd.read_csv(self.path + 'testset-levela.tsv', sep='\t', header=0)
        test_data_b = pd.read_csv(self.path + 'testset-levelb.tsv', sep='\t', header=0)
        test_data_c = pd.read_csv(self.path + 'testset-levelc.tsv', sep='\t', header=0)
        test_labels_a = pd.read_csv(self.path + 'labels-levela.csv', sep=',', header=None)
        test_labels_b = pd.read_csv(self.path + 'labels-levelb.csv', sep=',', header=None)
        test_labels_c = pd.read_csv(self.path + 'labels-levelc.csv', sep=',', header=None)

        # 清洗数据并进行词向量映射得到输入序列x
        print("Cleaning test data")
        test_data_a["clean data"] = test_data_a.tweet.apply(lambda x: clean_data(x)[0])
        sequences = self._tokenizer.texts_to_sequences(test_data_a["clean data"])
        self.test_a.x = pad_sequences(sequences, maxlen=self.max_seq_len)

        test_data_b["clean data"] = test_data_b.tweet.apply(lambda x: clean_data(x)[0])
        sequences = self._tokenizer.texts_to_sequences(test_data_b["clean data"])
        self.test_b.x = pad_sequences(sequences, maxlen=self.max_seq_len)

        test_data_c["clean data"] = test_data_c.tweet.apply(lambda x: clean_data(x)[0])
        sequences = self._tokenizer.texts_to_sequences(test_data_c["clean data"])
        self.test_c.x = pad_sequences(sequences, maxlen=self.max_seq_len)

        # 将字符标签对应到数字
        self.test_a.labels = np.array(test_labels_a[1].apply(lambda x: label_to_number_a[x]))
        self.test_b.labels = np.array(test_labels_b[1].apply(lambda x: label_to_number_b[x]))
        self.test_c.labels = np.array(test_labels_c[1].apply(lambda x: label_to_number_c[x]))
