import jieba
import gensim.models.word2vec as w2v
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec


def getw2v_20():
    # file = open('trainword.txt', encoding='utf-8')

    # cont = file.readline()
    # cont = cont.replace('，', ' ')
    # cont = cont.replace('。', ' ')
    # cont = cont.replace('“', ' ')
    # cont = cont.replace('”', ' ')
    # cont = cont.replace('？', ' ')
    # cont = cont.replace('：', ' ')
    # cont = cont.replace('‘', ' ')
    # cont = cont.replace('’', ' ')
    # cont = cont.replace('！', ' ')
    # cont = cont.replace('……', ' ')
    # cont = cont.replace('、', ' ')

    # seg_list = jieba.cut(cont)
    # sentences = ' '.join(seg_list)
    # sentences = sentences.split()[1:]
    # model = Word2Vec(sentences,size=20,iter=10, min_count=5, window=5, workers=2)
    # size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    # min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    # iter： 迭代次数，默认为5
    # window：表示当前词与预测词在一个句子中的最大距离是多少
    # workers参数控制训练的并行数。

    model_file_name = 'new_model_big.txt'
    # model_file_name = 'vec_model_new.model'
    # model.save(model_file_name)
    model = w2v.Word2Vec.load(model_file_name)
    return model


def getw2v_100():
    file = open('trainword.txt', encoding='utf-8')
    cont = file.readline()
    cont = cont.replace('，', ' ')
    cont = cont.replace('。', ' ')
    cont = cont.replace('“', ' ')
    cont = cont.replace('”', ' ')
    cont = cont.replace('？', ' ')
    cont = cont.replace('：', ' ')
    cont = cont.replace('‘', ' ')
    cont = cont.replace('’', ' ')
    cont = cont.replace('！', ' ')
    cont = cont.replace('……', ' ')
    cont = cont.replace('、', ' ')

    seg_list = jieba.cut(cont)
    sentences = ' '.join(seg_list)
    sentences = sentences.split()[1:]
    model = Word2Vec(sentences, size=100, iter=10, min_count=5, window=5, workers=2)

    # size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    # min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    # iter： 迭代次数，默认为5
    # window：表示当前词与预测词在一个句子中的最大距离是多少
    # workers参数控制训练的并行数。

    model_file_name = '100维度的词向量.txt'
    model.save(model_file_name)
    # model = w2v.Word2Vec.load(model_file_name)
    return model


def getw2v_30():
    file = open('trainword.txt', encoding='utf-8')
    cont = file.readline()
    cont = cont.replace('，', ' ')
    cont = cont.replace('。', ' ')
    cont = cont.replace('“', ' ')
    cont = cont.replace('”', ' ')
    cont = cont.replace('？', ' ')
    cont = cont.replace('：', ' ')
    cont = cont.replace('‘', ' ')
    cont = cont.replace('’', ' ')
    cont = cont.replace('！', ' ')
    cont = cont.replace('……', ' ')
    cont = cont.replace('、', ' ')

    seg_list = jieba.cut(cont)
    sentences = ' '.join(seg_list)
    sentences = sentences.split()[1:]
    model = Word2Vec(sentences, size=30, iter=10, min_count=5, window=5, workers=2)

    # size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    # min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    # iter： 迭代次数，默认为5
    # window：表示当前词与预测词在一个句子中的最大距离是多少
    # workers参数控制训练的并行数。

    model_file_name = '100维度的词向量.txt'
    model.save(model_file_name)
    # model = w2v.Word2Vec.load(model_file_name)
    return model


def getw2v_50():
    # file = open('trainword.txt', encoding='utf-8')
    # cont = file.readline()
    # cont = cont.replace('，', ' ')
    # cont = cont.replace('。', ' ')
    # cont = cont.replace('“', ' ')
    # cont = cont.replace('”', ' ')
    # cont = cont.replace('？', ' ')
    # cont = cont.replace('：', ' ')
    # cont = cont.replace('‘', ' ')
    # cont = cont.replace('’', ' ')
    # cont = cont.replace('！', ' ')
    # cont = cont.replace('……', ' ')
    # cont = cont.replace('、', ' ')

    # seg_list = jieba.cut(cont)
    # sentences = ' '.join(seg_list)
    # sentences = sentences.split()[1:]
    # model = Word2Vec(sentences,size=50,iter=10, min_count=5, window=5, workers=2)

    # size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    # min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    # iter： 迭代次数，默认为5
    # window：表示当前词与预测词在一个句子中的最大距离是多少
    # workers参数控制训练的并行数。

    model_file_name = 'word_embedding/50维度的词向量.txt'
    # model.save(model_file_name)
    model = w2v.Word2Vec.load(model_file_name)
    return model


def getw2v_100_sgns_weibo_bigram_char():
    filename = 'word_embedding/sgns.weibo.bigram-char'
    model = word2vec.Word2VecKeyedVectors.load_word2vec_format(filename, binary=False)
    return model


def Tencent_AILab_ChineseEmbedding():
    filename = 'word_embedding/Tencent_AILab_ChineseEmbedding.txt'
    model = word2vec.Word2VecKeyedVectors.load_word2vec_format(filename, binary=False)
    return model


# Tencent_AILab_ChineseEmbedding()

def getw2v_100_sgns_financial_bigram_char():
    filename = 'word_embedding/sgns.financial.bigram-char_cleaned'
    model = word2vec.Word2VecKeyedVectors.load_word2vec_format(filename, binary=False)
    # filename = 'sgns.financial.bigram-char'
    # model = word2vec.Word2VecKeyedVectors.load_word2vec_format(filename,binary = True)
    return model


def merge_sgns_bigram_char300():
    filename = 'word_embedding/merge_sgns_bigram_char300.txt'
    model = word2vec.Word2VecKeyedVectors.load_word2vec_format(filename, binary=False)
    # filename = 'sgns.financial.bigram-char'
    # model = word2vec.Word2VecKeyedVectors.load_word2vec_format(filename,binary = True)
    return model

# merge_sgns_bigram_char300()
