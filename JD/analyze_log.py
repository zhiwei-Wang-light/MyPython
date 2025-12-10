import os
import pandas as pd


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


def analyze_log_demo(work_dir, start_log, end_log):
    data = []

    log_list = os.listdir(work_dir)
    log_list = sorted(log_list)

    for log in log_list:
        if start_log<log<=end_log:
            log_path = os.path.join(work_dir, log)
            total_return = count_string_simple(log_path, "PackageRecover return")
            cage_wrong_return = count_string_simple(log_path, "PackageRecover return -6")

            error_rate = (cage_wrong_return / total_return * 100) if total_return > 0 else 0

            data.append({
                '日志文件': log,
                '返回次数': total_return,
                '播报笼车异常次数': cage_wrong_return,
            })

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 添加总计行
    total_row = {
        '日志文件': '总计',
        '总返回次数': df['返回次数'].sum(),
        '总播报笼车异常次数': df['播报笼车异常次数'].sum(),
        '错误率(%)': (df['播报笼车异常次数'].sum() / df['返回次数'].sum() * 100) if df[
                                                                                            '返回次数'].sum() > 0 else 0
    }

    # 显示表格
    pd.set_option('display.unicode.east_asian_width', True)
    print(df.to_string(index=False))

    print("\n" + "=" * 50)
    print(
        f"总计: 总返回次数={total_row['总返回次数']}, 总播报笼车异常次数={total_row['总播报笼车异常次数']}, 错误率={total_row['错误率(%)']:.2f}%")

    return df


if __name__ == '__main__':
    analyze_log_demo(
        "/home/jd/wangzhiwei225_data/cage_cart/test_data/tw_9_3/logs_20251124/home/jd/diskSpace/data/logs",
        "pack_vision_2025-11-19-05-45-02-279", "pack_vision_2025-11-24-19-01-29-780 ")
