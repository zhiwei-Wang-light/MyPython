import re
import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

def count_string_simple(file_path, target_string):
    """
    使用count方法统计字符串出现次数
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            count = content.count(target_string)
            return count
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return 0
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return 0


def parse_log_file(log_file_path):
    """
    解析日志文件，提取dist、ang和matrix数据
    只在"use dynamic coordinate"的下一行统计matrix

    Parameters:
    log_file_path: 日志文件路径

    Returns:
    dist_list: dist值列表
    ang_list: ang值列表
    matrix_list: 矩阵数据列表
    """
    dist_list = []
    ang_list = []
    matrix_list = []

    # 正则表达式模式
    dist_pattern = r'dist:([\d.-]+)'
    ang_pattern = r'ang:([\d.-]+)'
    matrix_pattern = r'\[matrix\]\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)'

    with open(log_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            # 提取dist和ang（从DynamicCoordinate行）
            if '[DynamicCoordinate]' in line:
                dist_match = re.search(dist_pattern, line)
                ang_match = re.search(ang_pattern, line)
                if dist_match and ang_match:
                    dist_list.append(float(dist_match.group(1)))
                    ang_list.append(float(ang_match.group(1)))

            # 检查是否是"use dynamic coordinate"行
            elif 'use dynamic coordinate' in line or 'dynamic coordinate system returns -6 when height greater than stack_top_thr_, use fixed coordinate' in line:
                # 检查下一行是否包含matrix
                if i + 1 < len(lines) and '[matrix]' in lines[i + 1]:
                    matrix_line = lines[i + 1]
                    matrix_match = re.search(matrix_pattern, matrix_line)
                    if matrix_match:
                        matrix_data = [float(matrix_match.group(i)) for i in range(1, 17)]
                        matrix_array = np.array(matrix_data).reshape(4, 4)
                        matrix_list.append(matrix_array)
                        i += 1  # 跳过下一行，因为已经处理了
            i += 1
    cage_wrong_return = count_string_simple(log_file_path, "fitness_score:1000")
    return dist_list, ang_list, matrix_list, cage_wrong_return


# 使用示例
if __name__ == "__main__":
    log_file = "/home/jd/wangzhiwei225_data/cage_cart/test_data/xm_xb_01/logs_1128/home/jd/diskSpace/data/logs/pack_vision_2025-11-27-22-25-13-541.txt"  # 替换为您的日志文件路径

    dist_list, ang_list, matrix_list, cage_wrong = parse_log_file(log_file)

    print(f"找到 {len(dist_list)} 个dist值")
    print(f"找到 {len(ang_list)} 个ang值")
    print(f"找到 {len(matrix_list)} 个矩阵")
    print(f"找到 {len(dist_list)-cage_wrong} 次成功")
    print(f"找到 {cage_wrong} 次失败")
    print(f"成功率: {(1-cage_wrong/len(dist_list))*100}%")
    succeed_dist=[d*1000 for d in dist_list if d<=0.1]
    succeed_ang = [a for a in ang_list if a != 1000]
    print(f"笼车平均位移量: {np.mean(succeed_dist)}mm")
    print(f"笼车平均旋转量: {np.mean(succeed_ang)}°")
    # 打印前几个值作为示例
    if dist_list:
        print("\n前5个dist值:", dist_list[:5])
    if ang_list:
        print("前5个ang值:", ang_list[:5])
    if matrix_list:
        print("第一个矩阵:")
        print(matrix_list[0])
    plt.figure(figsize=(10, 6))
    bax = brokenaxes(ylims=((0,100),))
    bax.plot(succeed_dist, marker=' ', linestyle='-', linewidth=1, label='trans')
    bax.standardize_ticks(xbase=10,ybase=10)
    # 添加标题和标签
    bax.set_title('Fluctuation')
    bax.set_xlabel('Frame', fontsize=12)
    bax.set_ylabel('mm', fontsize=12)

    # 添加图例
    bax.legend()

    # 添加网格
    bax.grid(True, linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()