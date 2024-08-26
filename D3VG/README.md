# D3VG
## 1.Environment
    torch==1.13.1+cuda==11.7 mmdeploy==0.12.0 transformers==4.35.2
## 2.Weights
#### Please create a checkpoints folder D3VG/checkpoints,and copy the downloaded weights to the folder
#### You can download required weights form this link:https://drive.google.com/drive/folders/1plOlm15jFwyiE8LH_6qQPZ-cmIWE6LQ6?usp=drive_link
## 3.Configuration method
#### You need to specify parameters for the following variables
    1.demo_for_en_640×480_onnx.py
        CAMERA_INTER      -------------------------------> the camera inter parameter of your depth
        SCALE_FACTOR      -------------------------------> the scanle factor parameter of your depth
#### You can deploy semantic segmentation models mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco through the [mmdeploy](https://github.com/open-mmlab/mmdeploy/tree/v0.12.0)
#### After you deploy mmdeploy, add object_detection_for_3d_visual_grounding.py to mmdeploy/demo/python/
#### In order to be able to make the positioning more accurate,you can choose other proper weights file from [mmdetection](https://github.com/open-mmlab/mmdetection/tree/v2.25.2)
## 4.Run demo
#### You can test your own data by running demo_for_en_640×480_onnx.py yourself,The size of your rgb data and depth data must be the same
## 5.train datasets distribution
#### The data we trained included the following categories, each with a different number of categories
    'person': 20553, 'chair': 35868, 'dining table': 13364, 'umbrella': 7663, 'box': 945, 'cabinet': 5106, 'bed': 5499,
    'picture': 1810, 'refrigerator': 2926, 'oven': 3081, 'sink': 4623, 'floor mat': 281, 'sofa': 7178, 'counter': 964,
     'vase': 4556, 'television': 5639, 'table': 4251, 'handbag': 7323, 'potted plant': 7015, 'suitcase': 5181, 'bottle': 10959,
      'clock': 4279, 'backpack': 5845, 'door': 3036, 'laptop': 4420, 'plastic bag': 999, 'trash can': 1222, 'toilet': 4293,
      'pillow': 2576, 'paper': 391, 'towel': 466, 'curtain': 784, 'clothes': 682, 'shelves': 1285, 'lamp': 501, 'window': 1821,
      'blinds': 347, 'dresser': 480, 'desk': 2985, 'shower curtain': 271, 'whiteboard': 433, 'microwave': 1513, 'bookshelf': 863,
      'bag': 392, 'night stand': 397, 'bathtub': 316, 'mirror': 325, 'toaster': 195
## 7.model deploy
#### We will provide the onnx file exported from the 3dvg model and provide the code for inference using the onnxruntime. 
#### The instance segmentation model can be deployed using [mmdeploy](https://github.com/open-mmlab/mmdeploy/blob/v0.12.0/docs/en/get_started.md),If your environment meets mmdeploy's requirements, you can do the following
    1.Convert Model
        python mmdeploy/tools/deploy.py \
        mmdeploy/configs/mmdet/instance-seg/instance-seg_tensorrt-fp16_dynamic-320x320-1344x1344.py \
        mmdetection/configs/mask_rcnn/mask_rcnn_r101_fpn_mstrain-poly_3x_coco.py \
        mask_rcnn_r101_fpn_mstrain-poly_3x_coco_20210524_200244-5675c317.pth \
        mmdetection/demo/demo.jpg \
        --work-dir mmdeploy_model/mask-rcnn \
        --device cuda \
        --dump-info
    2.object_detection_for_d3vg.py
        model_path     -------------------------------> mmdeploy_model/mask-rcnn



