# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import numpy as np
import cv2 as cv
from PIL import Image
from mmdeploy_python import Detector
def parse_args():
    parser = argparse.ArgumentParser(
        description='show how to use sdk python api')
    parser.add_argument('--device_name',default="cpu", help='name of device, cuda or cpu')
    parser.add_argument(
        '--model_path',default="D:\\DATA\\MyProjects\\Python\\D3VG\\segment_model\\mmdeploy_model",
        help='path of mmdeploy SDK model dumped by model converter')
    parser.add_argument('--image_path', default="D:\\DATA\\MyProjects\\Python\\D3VG\\segment_model\\mmdetection-2.25.2\\demo\\demo.jpg",help='path of an image')
    args = parser.parse_args()
    return args



args = parse_args()
detector = Detector(
    model_path=args.model_path, device_name=args.device_name, device_id=0)

# 设置打印选项
np.set_printoptions(threshold=np.inf)

def generate_seg(img, output_dir=None):

    img = cv.imread(img)

    bboxes, labels, masks = detector(img)
    img_h,img_w,_=img.shape
    indices = [i for i in range(len(bboxes))]
    masks_global=[]
    bboxes_global=[]
    labels_global=[]
    for index, bbox, label_id in zip(indices, bboxes, labels):
        [left, top, right, bottom], score = list(map(int,np.floor(bbox[0:4]))), bbox[4]
        if score < 0.5:
            continue

        if masks[index].size:
            mask = masks[index][:-1][:, :-1]
            blue, green, red = cv.split(img)
            zero_map = np.zeros((img_h, img_w))
            zero_map[top:top + mask.shape[0], left:left + mask.shape[1]] = mask
            zero_map=np.asarray(zero_map)==255
            masks_global.append(zero_map)
            bboxes_global.append([left, top, right, bottom])
            labels_global.append(labels[index])

            mask_img = blue[top:top + mask.shape[0], left:left + mask.shape[1]]
            cv.bitwise_or(mask, mask_img, mask_img)
            img = cv.merge([blue, green, red])
    image = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    image.show()
    return labels_global,masks_global,bboxes_global







if __name__ == '__main__':
    args = parse_args()
    generate_seg(args.image_path)
