import os
import cv2 as cv
import numpy as np


def sort_numeric_folders(path):
    # 获取所有文件夹
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    # 筛选纯数字文件夹
    numeric_folders = [f for f in folders if f.isdigit()]
    # 转换为数字并排序
    sorted_folders = sorted(numeric_folders, key=lambda x: int(x))
    return sorted_folders


def write_without_scientific_notation(fs, node_name, value):
    """写入数据时不使用科学计数法"""
    if isinstance(value, (int, float)):
        # 对于浮点数，使用format确保不使用科学计数法
        if isinstance(value, float):
            # 使用f-string格式化为字符串，确保不使用科学计数法
            fs.write(node_name, value)
        else:
            fs.write(node_name, value)
    else:
        fs.write(node_name, value)


def batch_process_config(cage_small_dir_path, key, new_value):
    node_list = ["P0", "Px", "Pxy", "size", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "offset_min",
                 "offset_max", "ros_type", "distance_for_fitcage", "distance_for_ishere", "angle_for_ishere", "guess"]

    mat_list = ["P0", "Px", "Pxy", "guess"]
    cage_small_root = cage_small_dir_path
    cage_small_lst = sort_numeric_folders(cage_small_dir_path)

    for cage_small in cage_small_lst:
        cage_small_path = os.path.join(cage_small_root, cage_small)
        fs = cv.FileStorage(cage_small_path + "/" + "template/config.xml", cv.FILE_STORAGE_READ)
        node_dict = {}
        for node in node_list:
            try:
                if node in mat_list:
                    node_dict[node] = fs.getNode(node).mat()
                else:
                    node_dict[node] = fs.getNode(node).real()
            except:
                pass
        fs.release()
        node_dict[key] = new_value

        print(f"pickzone{cage_small}修改后的配置为: ", node_dict)

        # 写入修改后的数据到原文件
        fs_write1 = cv.FileStorage(cage_small_path + "/" + "template/config.xml",
                                   cv.FILE_STORAGE_WRITE + cv.FILE_STORAGE_FORMAT_XML)
        for node in node_list:
            if node in node_dict:
                write_without_scientific_notation(fs_write1, node, node_dict[node])
        fs_write1.release()

        # 写入修改后的数据到另一个路径的文件
        target_path = cage_small_root + "/cage_small/" + cage_small + "/template/config.xml"
        # 确保目标路径的目录存在
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        fs_write2 = cv.FileStorage(target_path,
                                   cv.FILE_STORAGE_WRITE + cv.FILE_STORAGE_FORMAT_XML)
        for node in node_list:
            if node in node_dict:
                write_without_scientific_notation(fs_write2, node, node_dict[node])
        fs_write2.release()


if __name__ == "__main__":
    batch_process_config("/home/jd/RobotTaskSupervisor/PackVision/palletizingZones",
                         "ros_type",
                         1)