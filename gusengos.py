# !/usr/bin/python
# coding=utf-8

'''
分词
'''
import pandas as pd
from pandas import DataFrame
from zhon.hanzi import punctuation
from collections import Counter
import pynlpir
pynlpir.open()
pynlpir.nlpir.ImportUserDict(b'userdic.txt')

def seg_sentence(sentence):
    # 对句子进行分词
    sentence_seged = pynlpir.segment(sentence,pos_tagging=False)
    stopwords = stopwordslist('data/stopWord.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr
def stopwordslist(filepath):
    # 停用词
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


policy_seg = pd.read_csv('C://Users//86198//Desktop//学校//大三//专利信息检索与分析//data_all.csv',encoding='utf-8-sig')
policy_text = policy_seg["摘要 (中文)"].astype(str).tolist()

fenci_out = []
for i in range(len(policy_text)):
    line_seg = seg_sentence(policy_text[i])
    line_seg = line_seg.strip('0123456789')
    line_seg = line_seg.replace("\n", "")
    punctuation_str = punctuation
    for i in punctuation_str:
        line_seg = line_seg.replace(i, '')
    line_seg = ''.join([i for i in line_seg if not i.isdigit()])
    line_seg = line_seg.replace("(", "")
    line_seg = line_seg.replace(")", "")
    line_seg = line_seg.replace(".", "")
    line_seg = line_seg.replace("[", "")
    line_seg = line_seg.replace("]", "")
    line_seg = line_seg.replace("'", "")
    line_seg.replace("   ", " ")
    line_seg = line_seg.split(" ")
    counter = Counter(line_seg)
    dictionary = dict(counter)

    k = 300000000000
    res = counter.most_common(k)
    line_se = []

    for i in range(len(res)):
        if res[i][1] >= 0:
            line_se.append(res[i][0])
    line_s = []
    for word in line_seg:
        if word in line_se:
            line_s.append(word)
    while '' in line_s:
        line_s.remove('')
    fenci_out.append(line_s)
print(len(fenci_out))

ab = DataFrame(fenci_out)

f = open("result/result.txt", "w", encoding='utf-8-sig')
for l in fenci_out:
    f.write(str(l) + "\n")
f.close()


import time

import pandas as pd

"""
困惑度计算
"""
import gensim
from gensim import corpora
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel


def main():

    policy_seg = pd.read_csv('result/result.csv',encoding='utf-8-sig')
    f = policy_seg["key"].values.tolist()
    texts = [[word for word in line.split()] for line in f]
    M = len(texts)
    print('文本数目：%d 个' % M)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]  # 每个text对应的稀疏向量

    # 计算困惑度
    def perplexity(num_topics):
        ldamodel = LdaModel(corpus, num_topics=num_topics
                            , id2word=dictionary,
                            update_every=1, chunksize=400, passes=100,
                            iterations=200, random_state=1, minimum_probability=0.01)
        # corpus_tfidf, num_topics=num_topics, id2word=dictionary,
        #                           alpha=50/num_topics, eta=0.1, minimum_probability=0.001,
        #                           update_every=1, chunksize=100, passes=1

        # alpha=50/num_topics, eta=0.1
        # print(ldamodel.print_topics(num_topics=num_topics, num_words=15))
        # print(np.exp2(-(ldamodel.log_perplexity(corpus))))
        return np.exp2(-(ldamodel.log_perplexity(corpus)))

    '''
        print(np.exp2(-(ldamodel.log_perplexity(corpus))))
        return np.exp2(-(ldamodel.log_perplexity(corpus)))
    '''

    '''
    如果想要计算困惑度应该用：
    perplexity = np.exp2(-(ldamodel.log_perplexity())
    perplexity = 2**-(ldamodel.log_perplexity())#或者这个
    '''

    # 计算coherence
    def coherence(num_topics):
        ldamodel = LdaModel(corpus, num_topics=num_topics, alpha=50 / num_topics, eta=0.01,
                            id2word=dictionary, update_every=1, chunksize=400, passes=100,
                            iterations=400, random_state=1, minimum_probability=0.01)
        # print(ldamodel.print_topics(num_topics=num_topics, num_words=10))
        ldacm = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence='c_v')
        # print(ldacm.get_coherence())
        return ldacm.get_coherence()

    x = range(1, 30)  # 主题数目选择范围
    y = [perplexity(i) for i in x]  # 如果想用困惑度就选这个
    # y = [coherence(i) for i in x]
    plt.plot(x, y)
    plt.xlabel('主题数目')
    plt.ylabel('perplexity大小')
    # plt.ylabel('coherence大小')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.title('perplexity')
    plt.show()


if __name__ == '__main__':
    main()


'''
输出主题词
'''
import numpy as np
from gensim import corpora, models
from pandas.core.frame import DataFrame
import pyLDAvis.gensim_models

if __name__ == '__main__':

    num_topics = 7
    # 定义主题数
    tex1 = []
    f=open('result/result.txt','r',encoding='utf-8-sig')
    texts = [[word for word in line.split()] for line in f]
    M = len(texts)
    print('文本数目：%d 个' % M)

    dictionary = corpora.Dictionary(texts)
    V = len(dictionary)
    print('词的个数：%d 个' % V)

    corpus = [dictionary.doc2bow(text) for text in texts]  # 每个text对应的稀疏向量
    # TF-IDF
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    # LDA模型
    # corpus, num_topics=num_topics, alpha=50/num_topics, eta=0.1, id2word = dictionary, update_every=1, chunksize=400, passes=100,iterations=50
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                          alpha=50 / num_topics, eta=0.01,
                          minimum_probability=0.01,
                          update_every=1, chunksize=400, passes=100, random_state=1)
    # 政策 alpha=1, eta=0.1,  关于alpha与eta 大家可以自己进行调解
    # minimum_probability是概率低于此阈值的主题将被过滤掉。默认是0.01，设置为0则表示不丢弃任何主题。
    # 所有文档的主题
    doc_topic = [a for a in lda[corpus_tfidf]]
    # print('Document-Topic:')
    # print(doc_topic)
    doc_name = []
    doc_list = []
    doc_distrubute = []
    # 打印文档的主题分布
    num_show_topic = 1  # 每个文档显示前几个主题
    print('文档的主题分布：')
    doc_topics = lda.get_document_topics(corpus_tfidf)  # 所有文档的主题分布
    idx = np.arange(M)  # M为文本个数，生成从0开始到M-1的文本数组
    for i in idx:
        topic = np.array(doc_topics[i])
        topic_distribute = np.array(topic[:, 1])
        topic_idx = topic_distribute.argsort()[:-num_show_topic - 1:-1]  # 按照概率大小进行降序排列
        doc_name.append(i)
        doc_list.append(topic_idx)
        doc_distrubute.append(topic_distribute[topic_idx])
        print('第%d个文档的前%d个主题：' % (i, num_show_topic))
        print(topic_idx)
        print(topic_distribute[topic_idx])
    doc_topics_excel = {"文档名称": doc_name,
                        "主题": doc_list,
                        "概率": doc_distrubute}
    doc_excel = DataFrame(doc_topics_excel)  # 每个文档的主题概率
    doc_excel.to_excel('doc_topics_excel.xlsx')
    # 每个主题的词分布
    num_show_term = 15  # 每个主题显示几个词
    for topic_id in range(num_topics):
        print('主题#%d：\t' % topic_id)
        term_distribute_all = lda.get_topic_terms(topicid=topic_id)  # 所有词的词分布
        term_distribute = term_distribute_all[:num_show_term]  # 只显示前几个词
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:, 0].astype(np.int64)
        print('词：', end="")
        for t in term_id:
            print(dictionary.id2token[t], end=' ')
        print('概率：', end="")
        print(term_distribute[:, 1])
    # 将主题-词写入一个文档 topword.txt，每个主题显示20个词
    with open('topword.txt', 'w', encoding='utf-8') as tw:
        for topic_id in range(num_topics):
            term_distribute_all = lda.get_topic_terms(topicid=topic_id, topn=15)
            term_distribute = np.array(term_distribute_all)
            term_id = term_distribute[:, 0].astype(np.int64)
            for t in term_id:
                tw.write(dictionary.id2token[t] + " ")
            tw.write("\n")
    # lda 可视化
    d = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary, mds='mmds')
    pyLDAvis.save_html(d, 'lda_pass505.html')  # 可视化的图

'''
原本想搞主题演化 但报错
'''
# import numpy as np
# from scipy.spatial.distance import pdist
# from numpy import *
#
# import numpy as np
#
#
# def cosine_distance(vec1, vec2):
#     Vec = np.vstack([vec1, vec2])
#     dist2 = 1 - pdist(Vec, 'cosine')
#     return dist2
#
#
# # 余弦值计算方法
# def lda_cos(list1, list2):
#     a = []
#     b = []
#     for i in range(len(list1)):
#         for j in range(len(list2)):
#             if list1[i][0] == list2[j][0]:
#                 a.append(j)
#                 b.append(i)
#                 print(list1[i][0])
#                 print(list2[j][0])
#     for i in range(len(a)):
#         list2[a[i]], list2[b[i]] = list2[b[i]], list2[a[i]]
#     cos_dic1 = []
#     cos_dic2 = []
#     for i in range(len(list1)):
#         cos_dic1.append(list1[i][1])
#         cos_dic2.append(list2[i][1])
#
#     m = 1 - cosine_distance(cos_dic1, cos_dic2)
#     return m
#
#
# # 向量余弦值就算，在方法中已经将向量的互换方法写明，大家只需要将向量穿进去即可
# # 源数据如下：  list1 = ('a',0.11),('b',0.25）,('g',0.16)
# def tumple_1(list1, list2):
#     tuple_1 = []
#     for i in range(len(list1)):
#         a = [list1[i], list2[i]]
#         tup_t = tuple(a)
#         tuple_1.append(tup_t)
#     return tuple_1
#
#
# all = []
# for i in range(t1, t2, 10):
#     m = str(i)
#     data = []
#     sta = []
#     k = []
#     with open('topword.txt', encoding='utf8') as f:  # 这里为之前的lda生成的存入txt的主题向量
#         for line in f.readlines():
#             line = line.strip("\n")
#             line = line.split(",")
#             data.append(line)
#     for i in range(0, len(data), 2):
#         m = tumple_1(data[i], data[i + 1])
#         sta.append(m)
#     k.append(sta)
#     all.append(k)
# print(len(all))
# sim_all = []
# for m in range(len(all) - 1):
#     sim_lda35 = []
#     for i in range(len(all[m][0])):
#         ad = []
#         for j in range(len(all[m + 1][0])):
#             la = lda_cos(all[m][0][i], all[m + 1][0][j])
#             ad.append(la[0])
#         sim_lda35.append(ad)
#     sim_all.append(sim_lda35)
# print(sim_all)
# print('--------------------')
# m1 = []
# for i in range(len(sim_all)):
#     m2 = []
#     for j in range(len(sim_all[i])):
#         m3 = []
#         a = sum(sim_all[i][j])
#         for k in range(len(sim_all[i][j])):
#             m = sim_all[i][j][k] / a
#             m3.append(m)
#         m2.append(m3)
#     m1.append(m2)
# for i in range(len(m1)):
#     for j in range(len(m1[i])):
#         print(m1[i][j])
# print('----------')
# print(m1)
#
#

'''
数据去噪
'''
import pandas as pd
data=pd.read_csv('C:/Users/86198/Desktop/学校/大三/专利信息检索与分析/Knowledge_Graph_incoPat_1.csv',encoding='utf-8-sig')
data=data[data['摘要 (中文)'].apply(lambda x:True if '知识图谱' in str(x) else False )]
data.to_csv('C:/Users/86198/Desktop/学校/大三/专利信息检索与分析/Knowledge_Graph_new.csv', index=False,encoding='utf-8-sig')

'''
百度翻译 要钱
'''
import hashlib
import http
import json
import urllib
import random


def baiduTranslate(translate_text, flag=0):
    '''
    :param translate_text: 待翻译的句子，len(q)<2000
    :param flag: 1:原句子翻译成英文；0:原句子翻译成中文
    :return: 返回翻译结果。
    '''

    appid = '20230531001695594'
    secretKey = 'zWO8jVZQKIyeE_dx9iHP'
    httpClient = None
    myurl = '/api/trans/vip/translate'
    fromLang = 'auto'

    if flag:
        toLang = 'en'
    else:
        toLang = 'zh'

    salt = random.randint(3276, 65536)

    sign = appid + translate_text + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(translate_text) + '&from=' + fromLang + \
            '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign

    # 建立会话，返回结果
    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)

        # return result
        return result['trans_result'][0]['dst']

    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()


if __name__ == '__main__':

    data = pd.read_csv('C:/Users/86198/Desktop/学校/大三/科研方法与论文写作/New.csv', encoding='utf-8-sig')

    with open('C:/Users/86198/Desktop/学校/大三/科研方法与论文写作/translator.txt', 'w', encoding='utf-8-sig') as f:
        for index, row in data.iterrows():
            time.sleep(1)
            result = baiduTranslate(row['text'], flag=0)  # 百度翻译
            f.write(result+'\n')

            '''
            flag=1 输入的句子翻译成英文
            flag=0 输入的句子翻译成中文
            '''

'''
有道翻译 免费
'''
import requests
def translateYouDao(con):
    try:
        data = {'doctype': 'json',
                type: 'EN2ZH_CN',
                'i': con}
        r = requests.get("https://fanyi.youdao.com/translate", params=data)
        res_json = r.json()
        res_d = res_json['translateResult'][0]
        tgt = []
        for i in range(len(res_d)):
            tgt.append(res_d[i]['tgt'])
        return ''.join(tgt)
    except Exception as e:
        print('翻译失败：', e)
        return '翻译失败：' + con

# 因为我另外一门课也要用机器翻译
if __name__ == '__main__':
    data = pd.read_csv('C:/Users/86198/Desktop/学校/大三/科研方法与论文写作/New.csv', encoding='utf-8-sig')
    with open('C:/Users/86198/Desktop/学校/大三/科研方法与论文写作/translator.txt', 'w', encoding='utf-8-sig') as f:
        for index, row in data.iterrows():
            if 'nan' in row['text']:
                res = translateYouDao(str(row['text']))
                f.write(res+'\n')
