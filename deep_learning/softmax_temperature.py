import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
logits = np.random.randn(2, 55)


def softmax_with_temperature(x, temperature):
    tmp = np.exp(logits / temperature)
    return tmp / np.sum(tmp, axis=1, keepdims=True)


# 创建画布
plt.figure(figsize=(8, 6))
temperatures = [0.5, 1, 2]
# 绘制不同温度下的 Softmax 分布
for temperature in temperatures:
    probabilities = softmax_with_temperature(logits, temperature)
    plt.plot(range(len(logits[0])), probabilities[0], label=f'Temperature = {temperature}')

# 添加标题和标签
plt.title('Softmax with Different Temperature Values')
plt.xlabel('Classes')
plt.ylabel('Probability')
plt.xticks(range(len(logits[0])), [f'Class {i + 1}' for i in range(len(logits[0]))])
plt.legend()

# 显示图形
plt.grid(True)
plt.show()
