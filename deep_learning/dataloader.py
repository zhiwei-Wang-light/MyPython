# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        dataloader.py
# Author:           wzw
# Version:          0.1
# Created:          2024/12/14
# Description:      数据加载工具
# ------------------------------------------------------------------
import datasets
from datasets import load_dataset, Dataset
from torchvision import datasets as ds
from torchvision import transforms
import pandas as pd
import os
import json
import torch


class Dataload(object):
    def __int__(self):
        self.final_paths = []  # 初始化一个空列表用于存储路径

    def require_paths(self, input_path):
        """
        获得所有路径
        Args:
            input_path:

        Returns:

        """
        paths = []
        # 判断输入的路径是否是文件夹
        if os.path.isdir(input_path):
            # 如果是文件夹，遍历文件夹中所有的文件和子文件夹路径
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    paths.append(os.path.join(root, file))  # 将每个文件路径添加到final_paths中
        elif os.path.isfile(input_path):
            # 如果是文件，直接将文件路径添加到final_paths中
            paths.append(input_path)
        else:
            raise ValueError(f"输入的路径 '{input_path}' 既不是文件夹也不是文件。")
        return paths

    def load_csv(self, files_path: str, functions=None, test_size=0.1, encoding='ansi', remove_columns=None):
        """
        将csv数据转换为datasets格式
        Args:
            files_path: 路径,可以是文件也可以是文件夹
            functions: 自定义数据处理函数
            encoding: 编码格式
            test_size: 测试集比例
            remove_columns: 待删除列

        Returns:
            datasets格式
        """
        files_path = self.require_paths(files_path)
        # 读取所有CSV文件并合并
        dataframes = []
        # 遍历文件路径，读取每个CSV文件
        for file_path in files_path:
            try:
                # 使用适当的编码读取文件，假设编码为 windows-1252 (ANSI)
                df = pd.read_csv(file_path, encoding=encoding)
                dataframes.append(df)  # 将每个DataFrame添加到列表中
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        # 合并所有数据框
        merged_data = pd.concat(dataframes, ignore_index=True)
        dataset = Dataset.from_pandas(merged_data)
        dataset = dataset.train_test_split(test_size=test_size, seed=100)
        # 使用 map 函数修改数据集
        dataset = dataset.map(functions, remove_columns=remove_columns)
        return dataset

    def load_text(self, files_path: str, functions=None, test_size=0.1, remove_columns=None):
        """
        将txt数据转换为datasets格式
        Args:
            files_path: 路径,可以是文件也可以是文件夹
            functions: 自定义数据处理函数
            test_size: 测试集比例
            remove_columns: 待删除列

        Returns:
            datasets格式
        """
        files_path = self.require_paths(files_path)
        dataset = Dataset.from_text(files_path)
        dataset = dataset.train_test_split(test_size=test_size, seed=100)
        # 使用 map 函数修改数据集
        dataset = dataset.map(functions, remove_columns=remove_columns)
        return dataset

    def load_coco(self, train_image, train_ann, val_image, val_ann,batch_size=16, shuffle=True, num_workers=0):
        """
        将coco数据集加载为torch的dataset格式
        Args:
            train_image: coco文件夹路径
            train_ann: coco ann
            val_image: coco文件夹路径
            val_ann: coco ann
            batch_size: batch size
            shuffle: 是否打乱
            num_workers: 线程

        Returns:
            [train_loader, val_loader]
        """
        # 创建 coco dataset
        coco_train = ds.CocoDetection(train_image, train_ann)
        coco_val = ds.CocoDetection(val_image, val_ann)

        # 定义 coco collate_fn
        def collate_fn_coco(batch):
            return tuple(zip(*batch))

        # 创建 dataloader,
        train_loader = torch.utils.data.DataLoader(coco_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                                   pin_memory=True, collate_fn=collate_fn_coco, drop_last=True)
        val_loader = torch.utils.data.DataLoader(coco_val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                                                 collate_fn=collate_fn_coco, drop_last=False)
        return [train_loader, val_loader]
