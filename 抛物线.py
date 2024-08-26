import numpy as np
import matplotlib.pyplot as plt

# 定义抛物线函数
def parabola(w, a=1, b=0, c=0):
    return (a*w-c)**2

# 生成 w 值
w = np.linspace(-10, 10, 400)

# 计算 l 值
l = parabola(w, a=1, b=0, c=2)

# 绘制抛物线图
plt.plot(w, l, label='Parabola', color='blue')

# 添加标题和标签
plt.title('Parabola')
plt.xlabel('w')
plt.ylabel('l')

# 添加网格线
plt.grid(True)

# 显示图形
plt.legend()
plt.show()
