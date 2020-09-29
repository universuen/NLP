import jieba
import random
from gensim.models import Word2Vec


def get_texts(path):
    with open(path, encoding='utf-8') as f:
        texts = f.readlines()
    return texts

if __name__ == '__main__':
    # 获取语料库
    texts = get_texts("exp1_corpus.txt")

    # 使用jieba进行分词
    seqs_list = []
    for i in texts:
        seq = [word for word in jieba.cut(i)]
        seqs_list.append(seq)

    # 训练词向量
    model = Word2Vec(seqs_list, size=100, window=5, min_count=1, workers=4)

    # 使用词向量对指定词进行相关性比较
    print("相关性比较：" + model.similarity("中国", "中华"))

    # 寻找相似词
    print("相似词：" + model.wv.most_similar(positive=["武汉"], topn=5))

    # 寻找词类相似词
