import jieba
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


def get_texts(path):
    with open(path, encoding='utf-8') as f:
        texts = f.readlines()
    return texts

if __name__ == '__main__':
    try:
        model = Word2Vec.load("w2v.model")
    except Exception as e:
        print(e)
        # 获取语料库
        texts = get_texts("exp1_corpus.txt")
        # 使用jieba进行分词
        seqs_list = []
        for i in texts:
            seq = [word for word in jieba.cut(i)]
            seqs_list.append(seq)
        # 训练词向量
        model = Word2Vec(seqs_list, size=100, window=5, min_count=1, workers=4)
        # 保存模型
        model.save("w2v.model")

    # 使用词向量对指定词进行相关性比较
    print("相关性比较:")
    example1 = [
        ("中华", "中国"),
        ("习近平", "维尼")
    ]
    for i in example1:
        print(i, model.wv.similarity(i[0], i[1]))

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
            "positive": ["湖北", "郑州"],
            "negative": ["武汉"]
        },
        {
            "positive": ["辽宁", "济南"],
            "negative": ["沈阳"]
        }
    ]
    for i in example3:
        print("positive: %s" % i["positive"])
        print("negative: %s" % i["negative"])
        print(model.wv.most_similar(positive=i["positive"], negative=i["negative"], topn=5))

    print("********************")

    # 词向量降维与可视化
    example4 = ['江苏', '南京', '成都', '四川', '湖北', '武汉', '河南', '郑州', '甘肃', '兰州', '湖南', '长沙', '陕西', '西安',
                '吉林', '长春', '广东','广州', '浙江', '杭州']
    pca = PCA(n_components=2)
    embeddings = []
    for i in example4:
        embeddings.append(model.wv[i])
    results = pca.fit_transform(embeddings)
    for i, j in zip(example4, results):
        plt.annotate(i, j, family="Microsoft YaHei")
    sns.scatterplot(x=results[:, 0], y=results[:, 1])
    plt.show()
