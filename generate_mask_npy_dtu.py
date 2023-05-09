import numpy as np
from PIL import Image
from pathlib import Path
import os

root = '/home/yue/Desktop/monosdf/data/DTU/scan65'

mask_folder = Path(root, 'mask')

arr = os.listdir(mask_folder)
N = len(arr)

for i in range(N):
    img_path = Path(root, f'{i:06}_foreground_mask.png')
    img = Image.open(img_path).convert('L')
    img = np.array(img)
    img = img.astype(np.float32) / 255.0

    # save img to root as *.npy
    np.save(Path(root, f'{i:06}_mask.npy'), img)


