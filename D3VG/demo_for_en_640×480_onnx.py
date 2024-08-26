import argparse
import os
import time
from transformers import AutoTokenizer, MobileBertModel
import onnxruntime
import cv2 as cv
import numpy as np
import torch
from PIL import Image
from models.dataloader_vit_small_patch16_224_with_finetune import dataload, dataload_for_eval
from segment_model.mmdeploy.demo.python.object_detection_for_3d_visual_grounding import generate_seg
# envpath = '/home/light/anaconda3/envs/openmmlab/lib/python3.7/site-packages/cv2/qt/plugins/platforms'
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# -------- #
# 配置
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
# realsense
CAMERA_INTER = [389.911, 389.911, 321.213, 242.338]
# scaling factor
SCALE_FACTOR = 1000
def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "/" + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件，就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)
def uv2xyz(camera_inter, scalingFactor, depth, uv):
    fx, fy, centerX, centerY = camera_inter
    # -----------------------------------------------------------------
    # 像素坐标u，v
    # -----------------------------------------------------------------
    u, v = uv
    # -----------------------------------------------------------------
    # 相机坐标X，Y，Z
    # -----------------------------------------------------------------
    Z = depth.getpixel((u, v)) / scalingFactor
    if Z == 0 :
        return False
    else:
        X = (u - centerX) * Z / fx
        Y = (v - centerY) * Z / fy
        return [X, Y, Z]


def get_seg_info(rgb_path, depth_path, camera_inter, scale_factor):
    """
    分割rgb图像并将mask映射到三维
    Args:
        rgb_dir: 待分割的image文件夹路径
        depth_path: 深度图文件夹路径
        camera_inter: 相机内参
        scale_factor: 缩放因子

    Returns:
        dict: dtype=dict key: 相机id value: 第一个列表保存的segs分类,第二个列表保存mask在相机坐标系下的坐标,相机坐标系不是数组坐标系，好像是像素坐标系，第三个列表放入seg的box坐标
    """
    # 保存全部rgb的分割结果
    dict = {}
    # ---------------------------------------------------------------------
    # 第一个[]放入seg类别，第二个[]放入seg的mask坐标,第三个[]放入seg的box坐标
    # ---------------------------------------------------------------------
    dict["info"] = [[], [], [], []]
    color_image = cv.imread(rgb_image_path)
    t1 = time.time()
    boxes_class, segs, boxes = generate_seg(rgb_path)
    t2 = time.time()
    print("segment cost time:", t2 - t1)
    depth = Image.open(depth_path)
    for i, box_class in enumerate(boxes_class):
        # where返回的是numpy坐标，numpy坐标转化为像素坐标需要调换
        v, u = np.where(segs[i] == True)
        u = np.array((u), dtype=np.int32)
        v = np.array((v), dtype=np.int32)
        uvs = np.concatenate([u.reshape([u.shape[0], 1]), v.reshape([v.shape[0], 1])], axis=1)
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
def get_obb(points,i):
    """
    获取物体点云的中心坐标与8个顶点坐标的obb包围盒信息。

    Args:
        points (list): 物体点云的xyz坐标。

    Returns:
        tuple or None: 如果计算成功，返回一个包含中心坐标和顶点坐标的元组；如果出现异常，返回(None, None)。
    """
    try:
        # 处理点云数据
        pcd = o3d.geometry.PointCloud()
        if len(points) < 4:
            points = np.repeat(points, 4)

        pcd.points = o3d.utility.Vector3dVector(points)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.2)
        pcd = pcd.select_by_index(ind)

        # 计算obb包围盒
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
    对每个建议目标计算中心点
    :param dict_info:
    :param image:
    :return:
    """
    boxes = dict_info["info"][2]
    seg_points = dict_info["info"][1]
    rgb_points1 = dict_info["info"][3]
    # ----------------------- #
    # 截取region
    # ----------------------- #
    region_lst = []
    center_lst = []
    d3boxes = []
    bboxes = []
    ply_points = []
    rgb_points = []
    del_file("region/")
    for i, box in enumerate(boxes):
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
    每个图像分割出的建议目标数量是不一致的，在这里统一补齐到同一个长度
    :param batch_d3_patches:
    :return:
    """
    batch_d3_paded = []
    batch_d3_patches_token = []
    for b in batch_d3_patches:
        d3_patches = []
        for p in b:
            d3_patches.append([word2idx.index(round(float(i), 2)) for i in p])
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
    每个图像分割出的建议目标数量是不一致的，在这里统一补齐到同一个长度
    :param batch_image_patches:
    :return:
    """
    batches_features_list = []
    mask_list = []
    for image_patches in batch_image_patches:
        patches_list = []
        for image in image_patches:
            patches_list.append(image)
        patches_tensor = torch.stack(patches_list)
        patches_features_paded, padding_mask = patches_padding(args.max_length, patches_tensor)
        batches_features_list.append(patches_features_paded)
        mask_list.append(padding_mask)
    batches_features = torch.stack(batches_features_list)
    batch_mask = torch.stack(mask_list)
    return batches_features, batch_mask


def patches_padding(max_length, image_query):
    """
    Args:
        max_length: 最大长度
        image_qurey: 没有补齐的状态

    Returns:
        batch_image_qurey_paded: 补齐之后的序列，现在就可以当做是一个nlp来处理 shape=(b,l,dim)
        batch_image_qurey_padding_mask: 补齐之后的mask,shape=(b,l)
    """

    image_qurey_paded = torch.concat(
        (image_query, torch.zeros([(max_length - image_query.shape[0]), 384])), dim=0)
    padding_mask = torch.tensor(
        np.array([0] * image_query.shape[0] + [1] * (max_length - image_query.shape[0])),
        dtype=torch.float32)

    return image_qurey_paded, padding_mask


def predict(region_lst, center_lst, test_question):
    # 提取regions特征
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