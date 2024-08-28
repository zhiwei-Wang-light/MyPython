# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------
# File Name:        pth2onnx_mmdeploy_0.12.0.py
# Author:           wzw
# Version:          0.1
# Created:          2024/08/27
# Description:      转换openmmlab权重文件到onnx
# ------------------------------------------------------------------

from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK

img = '../D3VG/segment_model/mmpretrain-0.12.0/demo/demo.JPEG'
work_dir = '../D3VG/segment_model/mmdeploy_model'
save_file = 'end2end.onnx'
deploy_cfg = '../D3VG/segment_model/mmdeploy/configs/mmdet/instance-seg/instance-seg_onnxruntime_dynamic.py'
model_cfg = '../D3VG/segment_model/mmdetection-2.25.2/configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco.py'
model_checkpoint = '../D3VG/segment_model/mmdeploy_model/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth'
device = 'cpu'

# 1. convert model to onnx
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
  model_checkpoint, device)

# 2. extract pipeline info for sdk use (dump-info)
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)
