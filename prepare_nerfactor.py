import numpy as np
import json
import os
from pathlib import Path
from PIL import Image

exp = 'lego_3072'
data_dir = f"/home/yue/Desktop/dataset/nerfactor/dataset/{exp}/"
json_file = os.path.join(data_dir, "transforms_train.json")
output_dir = f"/home/yue/Desktop/dataset/nerfactor/dataset/monosdf/{exp}/"

image_out_dir = os.path.join(output_dir, "image")
mask_out_dir = os.path.join(output_dir, "mask")
Path(image_out_dir).mkdir(parents=True, exist_ok=True)
Path(mask_out_dir).mkdir(parents=True, exist_ok=True)

image_width = 512

with open(json_file) as f:
    data = json.load(f)

camera_angle_x = float(data["camera_angle_x"])

focal = 0.5 * image_width / np.tan(0.5 * camera_angle_x)



P = np.array([[-1*focal, 0, 256, 0],
[0, focal, 256, 0],
[0, 0, 1, 0],
 [0, 0, 0, 1]])

cameras = {}

for frame in data["frames"]:
    file_path = frame["file_path"]
    tokens = file_path.split("/")
    img_id = tokens[1].split("_")[1]
    i = int(img_id)
    norm_matrix = np.eye(4)
    r_t_matrix = np.linalg.inv(np.array(frame["transform_matrix"]))

    world_mat = np.matmul(P, r_t_matrix)

    cameras[f"world_mat_{i}"] = world_mat
    cameras[f"scale_mat_{i}"] = norm_matrix

    img = np.array(Image.open(os.path.join(data_dir, file_path + ".png")))


    mask = img[:,:,-1]
    # color = img[:,:,:-1]
    color = img[:,:,:-1].astype(np.float32) / 255.0
    _mask = mask.astype(np.float32) / 255.0
    color = color * np.expand_dims(_mask, axis=-1)
    color = (color * 255).astype(np.uint8)


    Image.fromarray(mask).save(os.path.join(mask_out_dir, f"{int(img_id):06}.png"))
    Image.fromarray(color).save(os.path.join(image_out_dir, f"{int(img_id):06}.png"))


np.savez(os.path.join(output_dir, "cameras.npz"), **cameras)