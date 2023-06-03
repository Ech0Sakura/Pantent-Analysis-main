# !/usr/bin/python
# encoding:utf-8

def get_Co_authors(filePath):

    with open(filePath, 'r', encoding='utf-8-sig') as f:
        text = f.read()
        co_authors_list = text.split('\n')
        co_authors_list.remove('')
        return co_authors_list

def str2csv(filePath, s):

    with open(filePath, 'w', encoding='utf-8-sig') as f:
        f.write(s)
    print('写入文件成功')

def sortDictValue(dict, is_reverse):

    tups = sorted(dict.items(), key=lambda item: item[1], reverse=is_reverse)
    s = ''
    for tup in tups:
        s = s + tup[0] + ',' + str(tup[1]) + '\n'
    return s

def build_matrix(co_keyword_list, is_reverse):

    node_dict = {}  # 节点字典
    edge_dict = {}  # 边字典

    for row_authors in co_keyword_list:
        row_authors_list = row_authors.split(',')

        for index, pre_au in enumerate(row_authors_list):

            if pre_au not in node_dict:
                node_dict[pre_au] = 1
            else:
                node_dict[pre_au] += 1

            if pre_au == row_authors_list[-1]:
                break
            connect_list = row_authors_list[index+1:]

            for next_au in connect_list:
                A, B = pre_au, next_au

                if A > B:
                    A, B = B, A
                key = A+','+B

                if key not in edge_dict:
                    edge_dict[key] = 1
                else:
                    edge_dict[key] += 1

    node_str = sortDictValue(node_dict, is_reverse)  # 节点
    edge_str = sortDictValue(edge_dict, is_reverse)   # 边
    return node_str, edge_str


if __name__ == '__main__':
    readfilePath = r'data_Keyword.csv'
    writefilePath1 = r'node.csv'
    writefilePath2 = r'edge.csv'

    co_keyword_list = get_Co_authors(readfilePath)

    node_str, edge_str = build_matrix(co_keyword_list, is_reverse=True)
    print(edge_str)

    str2csv(writefilePath1, node_str)
    str2csv(writefilePath2, edge_str)