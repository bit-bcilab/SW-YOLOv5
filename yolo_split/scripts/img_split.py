

import cv2
import math
import numpy as np
import shutil
from tqdm import tqdm
from glob import glob
from os.path import join
from multiprocessing import Pool
from utils.general import os, Path
from yolo_split.config import PATCH_SETTINGS, HYP, SCALE_SETTINGS

WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP
MODE = 'Visdrone'
# MODE = 'VISO'
# MODE = 'CarData'
PATCH_W, PATCH_H, NUM_PATCHES_W, NUM_PATCHES_H = PATCH_SETTINGS[MODE]
MAX_OCC, SHORT_THRESH = HYP[MODE]['max_occ'], HYP[MODE]['short_thresh']
overlap_w = (NUM_PATCHES_W * PATCH_W - 1.) / (NUM_PATCHES_W - 1.)
overlap_h = (NUM_PATCHES_H * PATCH_H - 1.) / (NUM_PATCHES_H - 1.)
MIN_W, MIN_H, MAX_W, MAX_H = SCALE_SETTINGS[MODE]


def _split(img, patch_w, patch_h, num_patches_w, num_patches_h):
    ih, iw = img.shape[:2]
    # num_patches = num_patches_w * num_patches_h
    size_w, size_h = int(iw * patch_w), int(ih * patch_h)

    interval_w = max(math.ceil((num_patches_w * size_w - iw) // (num_patches_w - 1)), 0)
    interval_h = max(math.ceil((num_patches_h * size_h - ih) // (num_patches_h - 1)), 0)
    patches_x1 = np.arange(0, num_patches_w) * (size_w - interval_w)
    patches_y1 = np.arange(0, num_patches_h) * (size_h - interval_h)
    patches_x1[-1] = iw - size_w
    patches_y1[-1] = ih - size_h
    patches_x2, patches_y2 = patches_x1 + size_w, patches_y1 + size_h
    bias_x, bias_y = np.meshgrid(patches_x1, patches_y1)
    bias = np.stack([bias_x, bias_y])

    sub_imgs = []
    for i in range(num_patches_w):
        for j in range(num_patches_h):
            sub_img = img[patches_y1[j]: patches_y2[j], patches_x1[i]: patches_x2[i]]
            sub_imgs.append(sub_img)

    return sub_imgs, bias


def center2corner(center, iw, ih):
    x, y, w, h = center
    x, w = x * iw, w * iw
    y, h = y * ih, h * ih
    x1 = x - w/2
    x2 = x + w/2
    y1 = y - h/2
    y2 = y + h/2
    corner = [x1, y1, x2, y2]
    return corner


def split(img, anno, patch_w, patch_h, num_patches_w, num_patches_h, with_ori=False):
    ih, iw = img.shape[:2]
    num_patches = num_patches_w * num_patches_h
    size_w, size_h = int(iw * patch_w), int(ih * patch_h)

    interval_w = max(math.ceil((num_patches_w * size_w - iw) // (num_patches_w - 1)), 0)
    interval_h = max(math.ceil((num_patches_h * size_h - ih) // (num_patches_h - 1)), 0)
    patches_x1 = np.arange(0, num_patches_w) * (size_w - interval_w)
    patches_y1 = np.arange(0, num_patches_h) * (size_h - interval_h)
    patches_x1[-1] = iw - size_w
    patches_y1[-1] = ih - size_h
    patches_x2, patches_y2 = patches_x1 + size_w, patches_y1 + size_h

    sub_annos = []
    sub_imgs = []
    for i in range(num_patches_w):
        for j in range(num_patches_h):
            sub_img = img[patches_y1[j]: patches_y2[j], patches_x1[i]: patches_x2[i]]
            sub_imgs.append(sub_img)
            sub_annos.append([])

    if anno is not None:
        num_objs = len(anno)
        for i in range(num_objs):
            label = anno[i].split()
            cls = label[0]
            box = label[1:]
            x, y, w, h = list(map(float, box))
            x, y, w_, h_ = x * iw, y * ih, w * iw / size_w, h * ih / size_h
            patch_xs = np.where((patches_x1 < x) & (patches_x2 > x))[0].tolist()
            patch_ys = np.where((patches_y1 < y) & (patches_y2 > y))[0].tolist()
            if len(patch_ys) and len(patch_xs):
                for patch_x in patch_xs:
                    for patch_y in patch_ys:
                        x_, y_ = (x - patches_x1[patch_x]) / size_w, (y - patches_y1[patch_y]) / size_h

                        x1, y1, x2, y2 = x_ - w_ / 2, y_ - h_ / 2, x_ + w_ / 2, y_ + h_ / 2
                        x1_, y1_, x2_, y2_ = max(0., x1), max(0., y1), min(1., x2), min(1., y2)

                        w, h, nw, nh = x2 - x1, y2 - y1, x2_ - x1_, y2_ - y1_
                        if (nw * nh > w * h * MAX_OCC) and min(nw / w, nh / h) > SHORT_THRESH:
                            nx, ny = (x1_ + x2_) / 2., (y1_ + y2_) / 2.
                            box_ = '{:d} {:8f} {:8f} {:8f} {:8f} \n'.format(int(cls), nx, ny, nw, nh)
                            if with_ori and ((nw > MAX_W) or (nh > MAX_H)):
                                continue
                            else:
                                a = 0
                            sub_annos[patch_x * num_patches_w + patch_y].append(box_)
                            a = 0
                        else:
                            a = 1
    # for i in range(num_patches):
    #     if sub_annos[i]:
    #         cv2.imwrite('.jpg', sub_imgs[i], [int(cv2.IMWRITE_JPEG_QUALITY), 1000])
    #         with open('.txt', 'w') as fl:
    #             fl.writelines(sub_annos[i])
    #             fl.close()
    return sub_imgs, sub_annos


def split_normal(root, sub):
    new_sub = sub + '-split'
    new_label_path = root / 'labels' / new_sub
    new_label_path.mkdir(parents=True, exist_ok=True)
    new_image_path = root / 'images' / new_sub
    new_image_path.mkdir(parents=True, exist_ok=True)

    annos = glob(join((root / 'labels' / sub), '*.txt'))
    annos = sorted(annos, key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))

    num_imgs = len(annos)
    for i in range(num_imgs):
        anno_path = annos[i]
        name = anno_path.split(os.sep)[-1].split('.')[0]
        img_name = name + '.jpg'
        img_path = root / 'images' / sub / img_name
        img = cv2.imread(str(img_path))

        if img is None:
            img_name = name + '.bmp'
            img_path = root / 'images' / sub / img_name
            img = cv2.imread(str(img_path))
            if img is None:
                continue
        with open(anno_path, 'r') as f:
            anno = f.readlines()
            f.close()
        sub_imgs, sub_annos = split(img, anno, 0.4, 0.4, 4, 4)

        for j in range(len(sub_imgs)):
            size_h, size_w = sub_imgs[0].shape[:2]
            if sub_annos[j]:
                # num_objs = len(sub_annos[j])
                # sub_img_ = sub_imgs[j].copy()
                # for n in range(num_objs):
                #     box = sub_annos[j][n].split()[1:]
                #     x, y, w, h = list(map(float, box))
                #     x, y, w, h = x * size_w, y * size_h, w * size_w, h * size_h
                #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
                #     box_ = [x1, y1, x2, y2]
                #     box_ = list(map(int, box_))
                #     cv2.rectangle(sub_img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
                # cv2.imshow('1', sub_img_)
                # cv2.waitKey(10)

                name_ = name + '-{:d}'.format(j)
                img_name_ = name_ + '.jpg'
                img_path_ = str(new_image_path / img_name_)
                cv2.imwrite(img_path_, sub_imgs[j], [int(cv2.IMWRITE_JPEG_QUALITY), 1000])
                anno_name_ = name_ + '.txt'
                anno_path_ = str(new_label_path / anno_name_)
                with open(anno_path_, 'w') as fl:
                    fl.writelines(sub_annos[j])
                    fl.close()


def split_visdrone(root, sub, with_ori=False):
    new_sub = sub + '-split'
    new_label_path = root / new_sub / 'labels'
    new_label_path.mkdir(parents=True, exist_ok=True)
    new_image_path = root / new_sub / 'images'
    new_image_path.mkdir(parents=True, exist_ok=True)

    annos = glob(join((root / sub / 'labels'), '*.txt'))
    # annos = sorted(annos, key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))

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
        sub_imgs, sub_annos = split(img, anno, PATCH_W, PATCH_H, NUM_PATCHES_W, NUM_PATCHES_H, with_ori=with_ori)

        size_h, size_w = sub_imgs[0].shape[:2]
        for j in range(len(sub_imgs)):
            if sub_annos[j]:
                # num_objs = len(sub_annos[j])
                # sub_img_ = sub_imgs[j].copy()
                # for n in range(num_objs):
                #     box = sub_annos[j][n].split()[1:]
                #     x, y, w, h = list(map(float, box))
                #     x, y, w, h = x * size_w, y * size_h, w * size_w, h * size_h
                #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
                #     box_ = [x1, y1, x2, y2]
                #     box_ = list(map(int, box_))
                #     cv2.rectangle(sub_img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
                # cv2.imshow('1', sub_img_)
                # cv2.waitKey()

                name_ = name + '-{:d}'.format(j)
                img_name_ = name_ + '.jpg'
                img_path_ = str(new_image_path / img_name_)
                cv2.imwrite(img_path_, sub_imgs[j], [int(cv2.IMWRITE_JPEG_QUALITY), 1000])
                anno_name_ = name_ + '.txt'
                anno_path_ = str(new_label_path / anno_name_)
                with open(anno_path_, 'w') as fl:
                    fl.writelines(sub_annos[j])
                    fl.close()

        if with_ori:
            filtering_anno = []

            if anno is not None:
                num_objs = len(anno)
                for i in range(num_objs):
                    label = anno[i].split()
                    cls = label[0]
                    box = label[1:]
                    x, y, w, h = list(map(float, box))
                    if w >= MIN_W or h >= MIN_H:
                        box_ = '{:d} {:8f} {:8f} {:8f} {:8f} \n'.format(int(cls), x, y, w, h)
                        filtering_anno.append(box_)
                    else:
                        a = 0
            if len(filtering_anno) > 0:
                shutil.copyfile(img_path, str(new_image_path / img_name))
                anno_name = name + '.txt'
                anno_path_ = str(new_label_path / anno_name)

                with open(anno_path_, 'w') as fl:
                    fl.writelines(filtering_anno)
                    fl.close()
            else:
                a = 0


def split_viso(root, sub):
    import random

    new_sub = sub + '-split'
    new_label_path = root / new_sub / 'labels'
    new_label_path.mkdir(parents=True, exist_ok=True)
    new_image_path = root / new_sub / 'images'
    new_image_path.mkdir(parents=True, exist_ok=True)

    annos = glob(join((root / sub / 'labels'), '*.txt'))
    annos = sorted(annos, key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))

    num_imgs = len(annos)
    for i in range(num_imgs):
        if i % 5 == 0:
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
            sub_imgs, sub_annos = split(img, anno, PATCH_W, PATCH_H, NUM_PATCHES_W, NUM_PATCHES_H)

            size_h, size_w = sub_imgs[0].shape[:2]
            for j in range(len(sub_imgs)):
                if sub_annos[j]:
                    # num_objs = len(sub_annos[j])
                    # sub_img_ = sub_imgs[j].copy()
                    # for n in range(num_objs):
                    #     box = sub_annos[j][n].split()[1:]
                    #     x, y, w, h = list(map(float, box))
                    #     x, y, w, h = x * size_w, y * size_h, w * size_w, h * size_h
                    #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
                    #     box_ = [x1, y1, x2, y2]
                    #     box_ = list(map(int, box_))
                    #     cv2.rectangle(sub_img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
                    # cv2.imshow('1', sub_img_)
                    # cv2.waitKey(10)

                    name_ = name + '-{:d}'.format(j)
                    img_name_ = name_ + '.jpg'
                    img_path_ = str(new_image_path / img_name_)
                    cv2.imwrite(img_path_, sub_imgs[j], [int(cv2.IMWRITE_JPEG_QUALITY), 1000])
                    anno_name_ = name_ + '.txt'
                    anno_path_ = str(new_label_path / anno_name_)
                    with open(anno_path_, 'w') as fl:
                        fl.writelines(sub_annos[j])
                        fl.close()


if __name__ == "__main__":
    # img = cv2.imread('F://DataBase/CarData/images/train/00140.jpg')
    # a = _split(img, 0.4, 0.4, 4, 4)

    # # size_w, size_h = 256, 256
    # # num_patches_w, num_patches_h = 6, 6
    # root = Path('F://DataBase/CarData/')
    # new_root = Path('F://DataBase/CarData-split/')
    # for sub in 'test', 'val':
    #     split_normal(root, sub)  # convert VisDrone annotations to YOLO labels

    root = Path('/home/Data2/Visdrone2019-DET/')  # D://DataBase
    for sub in ['VisDrone2019-DET-test-dev']:
        split_visdrone(root, sub, with_ori=True)  # convert VisDrone annotations to YOLO labels
