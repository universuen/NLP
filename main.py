import jieba
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
    print("相关性比较:")
    example1 = [
        ("中华", "中国"),
        ("武汉", "郑州")
    ]
    for i in example1:
        print(i, model.similarity(i))

    print("********************")

    # 寻找指定词相似词
    print("指定词相似词:")
    example2 = [
        ["武汉"],
        ["生活"]
    ]
    for i in example2:
        print(i, model.wv.most_similar(positive=i, topn=5))

    print("********************")

    # 寻找词类相似词
    print("词类相似词:")
    example3 = [
        {
            "positive": ["湖北", "成都"],
            "negative": ["武汉"]
        },
        {
            "positive": ["河南", "南京"],
            "negative": ["郑州"]
        }
    ]
    for i in example3:
        print("positive: %s" % i["positive"])
        print("negative: %s" % i["negative"])
        print(model.wv.most_similar(positive=i["positive"], negative=i["negative"], topn=5))

