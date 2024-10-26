from sklearn.feature_extraction.text import CountVectorizer

corpus = ["I come to China to travel",
          "This is a car polupar in China",
          "I love tea and Apple ",
          "The work is to write some papers in science"]
vectorizer = CountVectorizer()
print("词频统计：")
# 输出4个文本的词频统计：左边的括号中的两个数字分别为(文本序号，词序号)，右边数字为频次
print(vectorizer.fit_transform(corpus))
print("\n词袋模型：")
print(vectorizer.fit_transform(corpus).toarray())
