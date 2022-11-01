

import shutil
import cv2
import math
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import join
from multiprocessing import Pool
from utils.general import download, os, Path
from yolo_split.config import PATCH_SETTINGS, HYP

MODE = 'Visdrone'
# MODE = 'UAVDT'
PATCH_W, PATCH_H, NUM_PATCHES_W, NUM_PATCHES_H = PATCH_SETTINGS[MODE]
MAX_OCC, SHORT_THRESH = HYP[MODE]['max_occ'], HYP[MODE]['short_thresh']


def ignore_visdrone(root, sub):
    new_sub = sub + '-ignore'
    new_image_path = root / new_sub / 'images'
    # new_image_path.mkdir(parents=True, exist_ok=True)

    annos = glob(join((root / sub / 'annotations'), '*.txt'))

    num_imgs = len(annos)
    for i in range(num_imgs):
        anno_path = annos[i]
        name = anno_path.split(os.sep)[-1].split('.')[0]
        img_name = name + '.jpg'
        img_path = root / sub / 'images' / img_name
        img = cv2.imread(str(img_path))

        if img is None:
            img_name = name + '.bmp'
            img_path = root / sub / 'images' / img_name
            img = cv2.imread(str(img_path))
            if img is None:
                continue
        with open(anno_path, 'r') as f:
            anno = f.readlines()
            f.close()

        num_obj = len(anno)
        num_ignore = 0
        for j in range(num_obj):
            row = anno[j].split()[0].split(',')
            if row[4] == '0':
                num_ignore += 1
                x1, y1, w, h = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                x2, y2 = x1 + w, y1 + h
                img[y1: y2, x1: x2, :] = 114

        img_path_ = str(new_image_path / img_name)
        if num_ignore > 0:
            cv2.imwrite(str(img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 1000])
        # else:
        #     shutil.copyfile(img_path, img_path_)


if __name__ == "__main__":
    root = Path('/home/Data/Visdrone2019-DET/')
    for sub in ['VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']:
        ignore_visdrone(root, sub)  # convert VisDrone annotations to YOLO labels
