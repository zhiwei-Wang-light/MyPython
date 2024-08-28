import argparse
import os
import time
from transformers import AutoTokenizer, MobileBertModel
import onnxruntime
import cv2 as cv
import numpy as np
import torch
from PIL import Image
import open3d as o3d
from models.dataloader_vit_small_patch16_224_with_finetune import dataload, dataload_for_eval
from segment_model.mmdeploy.demo.python.object_detection_for_3d_visual_grounding import generate_seg
from utils.utils import uv2xyz,del_file
# -------- #
# config
# -------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--hidden_dim', default=512)
parser.add_argument('--dropout', default=0.1)
parser.add_argument('--nheads', default=4)
parser.add_argument('--dim_feedforward', default=2048)
parser.add_argument("--enc_layers", default=2)
parser.add_argument("--dec_layers", default=2)
parser.add_argument("--max_length", default=20)
parser.add_argument("--max_words_length", default=25)
parser.add_argument("--cls_num", default=20)
parser.add_argument("--pre_norm", default=True)
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(
            'checkpoints/mobilebert-uncased',local_files_only=True)
bert_model = MobileBertModel.from_pretrained("checkpoints/mobilebert-uncased",local_files_only=True).to(device)
for param in bert_model.parameters():
    param.requires_grad = False
bert_model.eval()
word2idx = [i / 100 for i in range(-100, 101)]
# realsense inter parameter
CAMERA_INTER = [389.911, 389.911, 321.213, 242.338]
# scaling factor
SCALE_FACTOR = 1000



def get_seg_info(rgb_path, depth_path, camera_inter:list, scale_factor):
    """
    Segment the image and transform it to the camera coordinate system
    Args:
        rgb_path: The path to the rgb image
        depth_path: The path to the depth image
        camera_inter: Camera inter parameter
        scale_factor: Scaling factor

    """
    dict = {}
    dict["info"] = [[], [], [], []]
    color_image = cv.imread(rgb_image_path)
    t1 = time.time()
    boxes_class, segs, boxes = generate_seg(rgb_path)
    t2 = time.time()
    print("segment cost time:", t2 - t1)
    depth = Image.open(depth_path)
    for i, box_class in enumerate(boxes_class):
        v, u = np.where(segs[i] == True)
        uvs = np.column_stack((u, v))
        dict["info"][0].append(box_class)
        dict["info"][2].append(boxes[i])
        xyz_list = []
        rgb_list = []
        for uv in uvs:
            xyz = uv2xyz(camera_inter, scale_factor, depth, uv)
            if xyz:
                bgr = color_image[int(uv[1]), int(uv[0])]
                rgb_list.append([bgr[2], bgr[1], bgr[0]])
                xyz_list.append(xyz)
        dict["info"][1].append(xyz_list)
        dict["info"][3].append(rgb_list)
    return dict
def get_obb(points:list,i):
    """
    Obtain the center coordinates of the object's point cloud and the OBB bounding box
    information with 8 vertex coordinates。

    Args:
        points (list): The XYZ coordinates of the object's point cloud。

    Returns:
        tuple or None: If the calculation is successful, return a tuple containing the center
        coordinates and vertex coordinates; if an exception occurs, return (None, None).。
    """
    try:
        pcd = o3d.geometry.PointCloud()
        if len(points) < 4:
            points = np.repeat(points, 4)

        pcd.points = o3d.utility.Vector3dVector(points)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.2)
        pcd = pcd.select_by_index(ind)
        obb = pcd.get_oriented_bounding_box()
        center = np.asarray(obb.get_center())
        vertex_set = np.asarray(obb.get_box_points())

        return center, vertex_set
    except Exception as e:
        print(f"Error in get_obb: {e}")
        return None, None

def data_info_extractor(rgb_image_path, depth_image_path):
    color_img = Image.open(rgb_image_path).convert('RGB')
    dict_info = get_seg_info(rgb_image_path, depth_image_path, CAMERA_INTER, SCALE_FACTOR)
    regions, centers, d3boxes, bboxes, ply_points, rgb_points = get_proposals(dict_info, color_img)
    return regions, centers, d3boxes, bboxes, ply_points, rgb_points


def get_proposals(dict_info, image):
    """
    Crop the target object in the image
    Args:
        dict_info: Information after segmenting the image
        image: Origin image

    Returns:

    """
    boxes = dict_info["info"][2]
    seg_points = dict_info["info"][1]
    rgb_points1 = dict_info["info"][3]
    # ----------------------- #
    # Crop region
    # ----------------------- #
    region_lst = []
    center_lst = []
    d3boxes = []
    bboxes = []
    ply_points = []
    rgb_points = []
    del_file("region/")
    for i, box in enumerate(boxes):
        # Crop the target object in the image
        region = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        center, vertex_set = get_obb(seg_points[i], i)
        if center != None:
            region.save("region/{}.jpg".format(i))
            region_lst.append(region)
            center_lst.append(center)
            d3boxes.append(vertex_set)
            bboxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            ply_points.append(seg_points[i])
            rgb_points.append(rgb_points1[i])
    return region_lst, center_lst, d3boxes, bboxes, ply_points, rgb_points


def d3_padding(batch_d3_patches):
    """
    The number of suggested targets segmented from each image is inconsistent;
    here, we standardize them to the same length
    :param batch_d3_patches:The 3D coordinates of the segmented objects shape=(b,n,3)
    :return: shape=(b,padding,3)
    """
    batch_d3_paded = []
    batch_d3_patches_token = []
    for d3s in batch_d3_patches:
        d3_patches = []
        for xyz in d3s:
            d3_patches.append([word2idx.index(round(float(i), 2)) for i in xyz])
        batch_d3_patches_token.append(d3_patches)
    for d3_patches in batch_d3_patches_token:
        d3_patches = np.asarray(d3_patches)
        d3_arrs_tensor_paded = torch.concat(
            (torch.from_numpy(d3_patches), torch.full([(args.max_length - d3_patches.shape[0]), 3], 201)), dim=0)
        batch_d3_paded.append(d3_arrs_tensor_paded)
    batch_d3_paded = torch.stack(batch_d3_paded)
    batch_d3_paded = torch.as_tensor(batch_d3_paded, dtype=torch.long)
    return batch_d3_paded


def image_padding(batch_image_patches):
    """
    The number of suggested targets segmented from each image is inconsistent;
     here, we standardize them to the same length
    :param batch_image_patches:The 3D coordinates of the segmented objects shape=(b,n,dim)
    :return:shape=(b,padding,dim)
    """
    batches_features_list = []
    mask_list = []
    for image_patches in batch_image_patches:
        patches_tensor = torch.stack(image_patches)  # Stack patches directly
        patches_features_paded, padding_mask = patches_padding(args.max_length, patches_tensor)
        batches_features_list.append(patches_features_paded)
        mask_list.append(padding_mask)
    batches_features = torch.stack(batches_features_list)
    batch_mask = torch.stack(mask_list)
    return batches_features, batch_mask


def patches_padding(max_length, image_query):
    """
    Args:
        max_length: max length
        image_qurey: The state without padding

    Returns:
        batch_image_qurey_paded: The padded sequence can now be treated as an NLP task, shape=(b,l,dim)
        batch_image_qurey_padding_mask: The padded mask,shape=(b,l)
    """

    image_qurey_paded = torch.concat(
        (image_query, torch.zeros([(max_length - image_query.shape[0]), 384])), dim=0)
    padding_mask = torch.tensor(
        np.array([0] * image_query.shape[0] + [1] * (max_length - image_query.shape[0])),
        dtype=torch.float32)

    return image_qurey_paded, padding_mask


def predict(region_lst, center_lst, test_question):
    # Extract image features of multiple object targets segmented from the complete
    # image using ViT (Vision Transformer).
    regions_feature = dataload_for_eval(region_lst)
    test_d3_patches = np.asarray([center_lst])
    test_d3_patches = test_d3_patches / np.max(abs(test_d3_patches))
    batch_d3 = d3_padding(
        test_d3_patches)
    batches_features, batch_mask = image_padding(
        [regions_feature])
    tgt_tokenized_text = tokenizer(
        test_question,
        add_special_tokens=True,
        max_length=args.max_words_length,
        padding='max_length', return_tensors="pt").to(device)
    tgt_attention_mask = 1.0 - tgt_tokenized_text['attention_mask'].clone().detach().to(device)
    outputs = bert_model(**tgt_tokenized_text)
    tgt_embedding = outputs.last_hidden_state.cpu().detach()
    batches_features = batches_features.cpu().detach().numpy()
    batch_mask = batch_mask.cpu().detach().numpy()
    batch_d3 = batch_d3.cpu().detach().numpy()
    tgt_embedding = tgt_embedding.numpy()
    tgt_attention_mask = tgt_attention_mask.cpu().detach().numpy()
    onnx_input = {'input1': batches_features, 'input2': batch_mask, 'input3': batch_d3, 'input4': tgt_embedding,
                  'input5': tgt_attention_mask}
    logits = onnx_model.run(None, onnx_input)
    return int(np.argmax(logits[0], axis=1))


device_name = 'cpu'  # or 'cpu'
if device_name == 'cpu':
    providers = ['CPUExecutionProvider']
elif device_name == 'cuda:0':
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

onnx_model = onnxruntime.InferenceSession('checkpoints/3dvg_small_script.onnx', providers=providers)


while True:
    print("please input rgb image:")
    rgb_image_path = input()
    print("please input depth image:")
    depth_image_path = input()
    depth = Image.open(depth_image_path)
    region_lst, center_lst, d3boxes, bboxes, ply_points, rgb_points = data_info_extractor(
        rgb_image_path,
        depth_image_path)
    while True:
        print("please input description:")
        description = input()
        if not description:
            break
        start_time = time.time()
        box_index = predict(region_lst, center_lst, [description])
        end_time = time.time()
        print("3dvg time cost:%fs" % (end_time - start_time))
        box_xyxy = bboxes[box_index]
        print("xyz of the target object is:",center_lst[box_index])
        image = cv.imread(rgb_image_path)
        image = cv.rectangle(image, box_xyxy[:2], box_xyxy[2:], color=(0, 255, 0), thickness=2)
        cv.namedWindow("output", cv.WINDOW_NORMAL)
        cv.imshow("output", image)
        if cv.waitKey(0) == 27:
            cv.destroyAllWindows()
# The potted plant to the left of the person with the white shirt