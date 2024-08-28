import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
device = "cuda" if torch.cuda.is_available() else "cpu"
image_processor = AutoImageProcessor.from_pretrained("checkpoints\\vit-small-patch16-224", local_files_only=True)
model = ViTModel.from_pretrained("checkpoints\\vit-small-patch16-224", local_files_only=True).to(device)
vit_small_patch16_224_checkpoints= "checkpoints\\pytorch_vit_small_patch16_224_with_finetune.pt"
model.load_state_dict(torch.load(vit_small_patch16_224_checkpoints,map_location=torch.device(device)))
model.eval()

def dataload(rootpath, scene_name=None):
    """
        Args:
            rootpath: The path that stores the proposed target and the proposed target center point for all scenes
                    .
                    |-- d3
                    |-- label
                    |-- region
            scene_name: If no scene name is specified, the features of the proposed target in all scenes are extracted

        Returns:
            all_scene_patches: Extract the features of the proposed target for all scenes
            all_d3_patches: Extract the center point of the proposed target for all scenes
    """
    train_region_floders_path = os.path.join(rootpath, 'region')
    train_d3_floders_path = os.path.join(rootpath, 'd3')
    region_floders_listdirs = os.listdir(train_region_floders_path)
    # -----------------------
    # Traversal folder
    # -----------------------
    all_scene_patches = {}
    for i, scene_dir in enumerate(region_floders_listdirs):
        if scene_name == '{}{}{}'.format(scene_dir[5:12], scene_dir[13:15], scene_dir[16:22]):
            image_patches = []
            in_listdirs = os.listdir(os.path.join(train_region_floders_path, scene_dir))
            # Sort by proposal id from smallest to largest
            in_listdirs.sort(key=lambda x: int(x.split('.')[0]))
            for in_dir in in_listdirs:
                image_path = os.path.join(os.path.join(train_region_floders_path, scene_dir), in_dir)
                image = Image.open(image_path)
                if image.mode!="RGB":
                    image=image.convert("RGB")
                image_patches.append(image)
            # Batch extracting features of proposed targets from a region folder
            inputs = image_processor(image_patches, return_tensors="pt").to(device)
            outputs = model(**inputs)
            image_patch_features = outputs.last_hidden_state[:, 0, :].cpu().detach()
            all_scene_patches['{}{}{}'.format(scene_dir[5:12], scene_dir[13:15], scene_dir[16:22])] = image_patch_features
            break
        if scene_name == None:
            image_patches = []
            in_listdirs = os.listdir(os.path.join(train_region_floders_path, scene_dir))
            # Sort by proposal id from smallest to largest
            in_listdirs.sort(key=lambda x: int(x.split('.')[0]))
            for in_dir in in_listdirs:
                image_path = os.path.join(os.path.join(train_region_floders_path, scene_dir), in_dir)
                image = Image.open(image_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image_patches.append(image)
            # Batch extracting features of proposed targets from a region folder
            inputs = image_processor(image_patches, return_tensors="pt").to(device)
            outputs = model(**inputs)
            image_patch_features = outputs.last_hidden_state[:, 0, :].cpu().detach()
            all_scene_patches['{}{}{}'.format(scene_dir[5:12], scene_dir[13:15], scene_dir[16:22])] = image_patch_features
    # --------------------- #
    # Extract the 3d coordinates corresponding to the images in each folder
    # --------------------- #
    all_d3_patches = {}
    d3_out_listdirs = os.listdir(train_d3_floders_path)
    for scene_dir in d3_out_listdirs:
        if scene_name == '{}{}{}'.format(scene_dir.split('.')[0][5:12], scene_dir.split('.')[0][13:15],
                                         scene_dir.split('.')[0][16:22]):
            d3_txt = os.path.join(train_d3_floders_path, scene_dir)
            with open(d3_txt, mode="r") as r:
                lines = r.readlines()
                d3_lst = []
                for line in lines:
                    d3_lst.append(list(map(float, line.strip().split())))
                r.close()
            all_d3_patches['{}{}{}'.format(scene_dir.split('.')[0][5:12], scene_dir.split('.')[0][13:15],
                                           scene_dir.split('.')[0][16:22])] = d3_lst
            break
        if scene_name == None:
            d3_txt = os.path.join(train_d3_floders_path, scene_dir)
            with open(d3_txt, mode="r") as r:
                lines = r.readlines()
                d3_lst = []
                for line in lines:
                    d3_lst.append(list(map(float, line.strip().split())))
                r.close()
            all_d3_patches['{}{}{}'.format(scene_dir.split('.')[0][5:12], scene_dir.split('.')[0][13:15],
                                           scene_dir.split('.')[0][16:22])] = d3_lst
    return all_scene_patches, all_d3_patches


def dataload_for_eval(region_lst):
    """
        Used for prediction or evaluation
        Args: region_lst: A list of proposed targets
        return: The features of the proposed targets
    """
    inputs = image_processor(region_lst, return_tensors="pt").to(device)
    outputs = model(**inputs)
    image_patch_features = outputs.last_hidden_state[:, 0, :].cpu().detach()
    return image_patch_features
