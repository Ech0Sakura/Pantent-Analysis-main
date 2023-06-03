# !/usr/bin/python
# coding=utf-8

'''
边列表构建
'''
import pandas as pd
data=pd.read_csv('technology_effect.csv',encoding='utf-8-sig',engine='python')
with open('cooperate.txt','w',encoding='utf-8-sig') as file:
    for index, row in data.iterrows():
        for i in range(1,12,1):
            if 'nan' not in str(row['Keyword'+str(i)]):
                file.write(str(row['Keyword'+str(i)])+ '\n')

file.close()

'''
去重
'''
import pandas as pd
frame=pd.read_csv('cooperate.csv',encoding='utf-8-sig',engine='python')
data = frame.drop_duplicates(subset=['country'], keep='first', inplace=False)
data.to_csv('new1.csv', encoding='utf-8-sig')

'''
这个不算，是为了构建检索式的
'''
# import pandas as pd
# frame=pd.read_csv('C://Users//86198//Desktop//学校//大三//专利信息检索与分析//data_all.csv',encoding='utf-8-sig',engine='python')
# with open('cooperate.txt','w',encoding='utf-8-sig') as file:
#     for index, row in frame.iterrows():
#         file.write('IPC=('+row['公开（公告）号']+') OR ')

'''
流向图数据
'''
import pandas as pd
data=pd.read_csv('new1.csv',encoding='utf-8-sig',engine='python')
with open('cooperate.txt','w',encoding='utf-8-sig') as file:
    for index, row in data.iterrows():
        # ["Goroka","Goroka","Papua New Guinea",145.391881,-6.081689],
        # name	city
        file.write('["'+str(row['name'])+'","'+str(row['city'])+'","'+str(row['country'])+'",'+str(row['longitude'])+','+str(row['latitude'])+'],')

'''
H-index 计算 没用到
'''
import pandas as pd
def Hindex(indexList):
    indexSet = sorted(list(set(indexList)), reverse = True)
    for index in indexSet:
        #clist为大于等于指定引用次数index的文章列表
        clist = [i for i in indexList if i >=index ]
        #由于引用次数index逆序排列，当index<=文章数量len(clist)时，得到H指数
        if index <=len(clist):
            break
    return index

if __name__ == '__main__':
    data=pd.read_csv('new3.csv',encoding='utf-8-sig',engine='python')
    group=data.groupby('inventor')
    with open('cooperate.txt','w',encoding='utf-8-sig') as file:
        for index, grouped in group:
            file.write(str(index)+'\t'+str(Hindex(grouped['count']))+'\n')
            print(index)
            print(Hindex(grouped['count']))

'''
原本打算提取SAO结构，但是太费时间了
'''
import spacy
import pandas as pd
nlp = spacy.load("zh_core_web_sm")
data=pd.read_csv('C://Users//86198//Desktop//学校//大三//专利信息检索与分析//ca.csv',encoding='utf-8-sig',engine='python')
with open('cooperation.txt','w',encoding='utf-8-sig') as file:

    for index, row in data.iterrows():
        doc = nlp(str(row['标题 (中文)']))
        for token in doc:
            a="{0}/{1} <--{2}-- {3}/{4}".format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_)
            if 'compound' in token.dep_:
                file.write(str(row['标题 (中文)'])+'\t'+a+'\n')

#
'''
通过稀疏矩阵构建邻接矩阵 但失败了
'''
# import scipy.sparse as sp
# import pandas as pd
# import networkx as nx
#
#
# df = pd.read_table('cooperation.txt', sep=' ', header=None) # 读取数据集为 pandas
# relation_df = pd.DataFrame(df, columns=[0, 1]) # 取出交互关系
#
#
# relation_list = []
# for index, row in relation_df.iterrows():
#     relation_list.append((row[0], row[1]))
#
# g = nx.Graph(relation_list)
# d_A = nx.to_numpy_matrix(g)
#
# s_A = sp.csr_matrix(d_A) # numpy 转换为 稀疏矩阵
# sp.save_npz('adj.npz', s_A)  # 保存稀疏矩阵
# import numpy
# import scipy.sparse as sp
#
# csr_matrix_variable = sp.load_npz('adj.npz') # 读取稀疏矩阵
# numpy.savetxt('new.csv', csr_matrix_variable.todense(), delimiter = '\t')


'''
这个不算，是信息计量学的边列表构建
但是我电脑没空间了 所以写这里
'''
# import pandas as pd
# import numpy as np
#
# # 导入你的数据
# data = pd.read_csv('cooperation.csv',encoding='utf-8-sig')
#
# vals = np.unique(data[['co1', 'co2']])  # 同时取出两列,作为节点
# df = pd.DataFrame(0, index=vals, columns=vals)
# f = df.index.get_indexer
# df.values[f(data.co1), f(data.co2)] = 1
# df.to_csv('new.csv',encoding='utf-8-sig')
#
# f = open('C://Users//86198//Desktop//学校//大三//信息分析与预测//内容分析法//cnki all.txt','r',encoding='utf-8-sig')
# with open('cooperate.txt','w',encoding='utf-8-sig') as file:
#     for lines in f:
#         if 'Keyword-关键词' in lines:
#             ls = lines.replace('Keyword-关键词:  ', '').replace('\n','').split(';')
#             for i in ls:
#                 file.write(i+',')
#
#             file.write('\n')
#
# f.close()
# file.close()