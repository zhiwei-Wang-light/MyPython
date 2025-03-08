# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        naive_bayes.py
# Author:           wzw
# Version:          0.1
# Created:          2024/10/26
# Description:      朴素贝叶斯分类,连续特征可以使用高斯朴素贝叶斯,P(A|B)=P(B|A)*P(A)/P(B),P(B|A)特征之间条件独立,P(A)为类别出现的概率
# ------------------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 创建朴素贝叶斯分类器（高斯朴素贝叶斯）
gnb = GaussianNB()

# 训练模型
gnb.fit(X_train, y_train)

# 进行预测
y_pred = gnb.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# 输出结果
print(f"准确率: {accuracy:.4f}")
print("混淆矩阵:")
print(conf_matrix)
print("分类报告:")
print(class_report)
