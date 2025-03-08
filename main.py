# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from leetcode.algo import topKFrequent
from deep_learning.dataloader import Dataload

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nums = [1, 1, 1, 2, 2, 3, 5, 5, 5, 4, 6, 6, 9, 9, 9, 9]
    print("前k个高频元素", topKFrequent(nums, 2))
    dataload=Dataload()
    dataload.load_coco("F:/数据集/OSCD/coco_carton/oneclass_carton/images/train2017",
                       "F:/数据集/OSCD/coco_carton/oneclass_carton/annotations/instances_train2017.json",
                       "F:/数据集/OSCD/coco_carton/oneclass_carton/images/val2017",
                       "F:/数据集/OSCD/coco_carton/oneclass_carton/annotations/instances_val2017.json"

                       )
