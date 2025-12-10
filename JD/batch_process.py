import os
import cv2 as cv


def sort_numeric_folders(path):
    # 获取所有文件夹
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    # 筛选纯数字文件夹
    numeric_folders = [f for f in folders if f.isdigit()]
    # 转换为数字并排序
    sorted_folders = sorted(numeric_folders, key=lambda x: int(x))
    return sorted_folders


def batch_process_config(cage_small_dir_path, key, new_value):
    node_list = ["P0", "Px", "Pxy", "size", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "offset_min",
                 "offset_max", "distance_for_fitcage", "distance_for_ishere", "angle_for_ishere", "guess"]

    mat_list = ["P0", "Px", "Pxy", "guess"]
    cage_small_root = cage_small_dir_path
    cage_small_lst = sort_numeric_folders(cage_small_dir_path)
    for cage_small in cage_small_lst:
        cage_small_path = os.path.join(cage_small_root, cage_small)
        fs = cv.FileStorage(cage_small_path + "/" + "template/config.xml", cv.FILE_STORAGE_READ)
        node_dict = {}
        for node in node_list:
            if node in mat_list:
                node_dict[node] = fs.getNode(node).mat()
            else:
                node_dict[node] = fs.getNode(node).real()
        fs.release()
        node_dict[key] = new_value

        print(f"pickzone{cage_small}修改后的配置为: ", node_dict)
        # 写入修改后的数据到新文件
        fs_write1 = cv.FileStorage(cage_small_path + "/" + "template/config.xml", cv.FILE_STORAGE_WRITE)
        for node in node_list:
            fs_write1.write(node, node_dict[node])
        fs_write1.release()
        # 写入修改后的数据到新文件
        fs_write2 = cv.FileStorage(cage_small_root + "/cage_small/" + cage_small + "/template/config.xml",
                                   cv.FILE_STORAGE_WRITE)
        for node in node_list:
            fs_write2.write(node, node_dict[node])
        fs_write2.release()


if __name__ == "__main__":
    batch_process_config("/home/jd/wangzhiwei225/JDCode/RobotTaskSupervisor/PackVision/palletizingZones",
                         "distance_for_ishere",
                         0.1)
