# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        tf_idf.py
# Author:           wzw
# Version:          0.1
# Created:          2024/10/26
# Description:      词频-逆文档频率,词频即单词出现在该文中的频率,逆文档频率即log(文档总数/包含该单词的文档数+1),可以用作关键词排序,
#                   也可以用作文本分类,文本相似性计算
# ------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

documents = ["机器学习是人工智能的一个分支",
             "深度学习是机器学习的一个子领域",
             "自然语言处理是机器学习的一个应用"]


def chinese_tokenizer(text):
    return list(jieba.cut(text))


# 如果对象是中文,需要自定义分词器
tfidf_vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
# 全部特征名称
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_array = tfidf_matrix.toarray()
for i in range(len(documents)):
    print(f"文档{i + 1}的TF-IDF值为:")
    for j in range(len(feature_names)):
        if tfidf_array[i][j] > 0:
            print(f"{feature_names[j]}:{tfidf_array[i][j]:.4f}")
