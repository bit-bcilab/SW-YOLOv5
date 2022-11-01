

import glob
import jpeg4py as jpeg
import cv2
from PIL import Image
import math
import random
import numpy as np
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, distributed

from utils.datasets import IMG_FORMATS, VID_FORMATS
from utils.general import Path, os, LOGGER, xywhn2xyxy, xyxy2xywhn, colorstr, clip_coords
from utils.augmentations import letterbox, random_perspective, augment_hsv
from utils.torch_utils import torch_distributed_zero_first
from yolo_split.config import PATCH_SETTINGS, ALBUMENT, HYP, set_attrs, SCALE_SETTINGS, SCALE_SETTINGS_VAL

WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP
READ_WAY = 'jpeg'
MODE = 'Visdrone'
# MODE = 'UAVDT'
PATCH_W, PATCH_H, NUM_PATCHES_W, NUM_PATCHES_H = PATCH_SETTINGS[MODE]
overlap_w = (NUM_PATCHES_W * PATCH_W - 1.) / (NUM_PATCHES_W - 1.)
overlap_h = (NUM_PATCHES_H * PATCH_H - 1.) / (NUM_PATCHES_H - 1.)
MIN_W, MIN_H, MAX_W, MAX_H = SCALE_SETTINGS[MODE]
MIN_W = overlap_w if MIN_W > overlap_w else MIN_W
MIN_H = overlap_h if MIN_H > overlap_h else MIN_H
MAX_OCC, SHORT_THRESH = HYP[MODE]['max_occ'], HYP[MODE]['short_thresh']


def read_img(img_path, way='opencv', read_bgr=False):

    if way == 'opencv':
        img = cv2.imread(img_path)
        if not read_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif way == 'PIL':
        img = np.array(Image.open(img_path))
        if read_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    elif way == 'jpeg':
        if img_path.endswith('.jpg') or img_path.endswith('.JPEG'):
            try:
                img = jpeg.JPEG(img_path).decode()
                if read_bgr:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except:
                img = read_img(img_path, way='opencv')
        else:
            img = read_img(img_path, way='opencv')
    return img


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            self.transform = ALBUMENT[MODE]

            LOGGER.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(colorstr('albumentations: ') + f'{e}')

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


class Generator(Dataset):
    def __init__(self, root, sub,
                 img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='',
                 patch_w=None, patch_h=None, num_patches_w=None, num_patches_h=None):
        with open(str(root / (sub + '.json')), 'r') as f:
            data = json.load(f)
            f.close()
        if os.path.isdir(root / sub / 'images'):
            self.sub = root / sub / 'images'
        elif os.path.isdir(root / 'images' / sub):
            self.sub = root / 'images' / sub

        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.stride = stride
        self.albumentations = Albumentations() if augment else None

        self.ori_prob = 0.1
        if patch_w is None:
            patch_w = PATCH_W
        if patch_h is None:
            patch_h = PATCH_H
        if num_patches_w is None:
            num_patches_w = NUM_PATCHES_W
        if num_patches_h is None:
            num_patches_h = NUM_PATCHES_H
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.num_patches_w = num_patches_w
        self.num_patches_h = num_patches_h
        self.num_patches = self.num_patches_w * self.num_patches_h
        self.num_rate = 10  # 一个epoch中训练的step数等于 单张图片数 * 每张图像被训练的次数

        self.protect_rate = (-1.0, 0.10)  # 取值范围。若取-0.75，代表选中物体周围1.75倍边长的范围必须保留在crop区域内
        self.max_occ = 0.47  # 若crop后物体剩余面积不到原来的0.65，则从标签中去掉
        self.short_thresh = 0.54

        self.scale_prob = 0.7  # patch尺寸缩放的概率
        self.scale_jitter = 1.3  # 最大尺寸抖动倍率。即裁剪尺寸最多可放大到预设尺寸的1.3倍, 或缩小到预设尺寸的1/1.3

        # 各项马赛克扩增的概率
        self.mosaic_vertical_prob = 0.08
        self.mosaic_horizontal_prob = 0.16
        self.mosaic_embed_prob = 0.25
        self.mosaic_diag_r_prob = 0.35
        self.mosaic_diag_l_prob = 0.4
        self.mosaic_3_prob = .55
        self.mosaic_4_prob = 0.7
        self.embed_rate = (0.30, 0.60)  # 取值范围。需要裁剪出更小的patch，以进行嵌入粘贴。若取0.5，且patch_w为0.4，代表裁剪时的尺寸为0.4*0.5=0.2

        self.copy_paste_prob = 0.60  # 复制粘贴扩增的概率
        self.easy_paste_prob, self.similar_paste_prob, self.cluster_paste_prob = 0.7, 0.25, 0.5

        self.perspective_prob = 0.4
        self.hsv_prob = 0.15

        sw_hyp = HYP[MODE]
        sw_hyp['MODE'] = MODE
        sw_hyp['PATCH_SETTINGS'] = PATCH_SETTINGS[MODE]
        sw_hyp['SCALE_SETTINGS'] = SCALE_SETTINGS[MODE]
        sw_hyp['SCALE_SETTINGS_VAL'] = SCALE_SETTINGS_VAL[MODE]
        self.sw_hyp = sw_hyp

        set_attrs(self, HYP[MODE])
        self.mosaic_prob_list = np.array(
            [self.mosaic_vertical_prob, self.mosaic_horizontal_prob, self.mosaic_embed_prob,
             self.mosaic_diag_r_prob, self.mosaic_diag_l_prob, self.mosaic_3_prob, self.mosaic_4_prob])
        self.mosaic_func_list = [self.mosaic2_vertical, self.mosaic2_horizontal, self.mosaic2_embed,
                                 self.mosaic2_diag_r, self.mosaic2_diag_l, self.mosaic3, self.mosaic4]
        if not augment:
            self.scale_prob = 0.
            self.scale_jitter = 1.

        # 建立粘贴objs lib
        with open(str(root / (sub + '-cls.json')), 'r') as f:
            obj_data = json.load(f)
            self.cls_dict = obj_data['cls_dict']
            self.num_cls = len(self.cls_dict)
            self.obj_num = obj_data['cls_num']
            self.obj_imgs = obj_data['imgs']
            self.objs = obj_data['objs']
            if MODE == 'VisDrone':
                # visdrone
                self.ban_cls = [0, 1, 2, 6, 7, 9]
                self.valid_cls = [3, 4, 5, 8]
            elif MODE == 'UAVDT':
                # uavdt
                self.ban_cls = []
                self.valid_cls = [0, 1, 2]
            f.close()

        self.imgs = data['imgs']
        self.num_imgs = len(self.imgs)
        self.annos = data['labels']
        self.annos_abs = data['labels_abs']

        sw_shapes = data['shapes'].copy()
        sw_shapes = np.array(sw_shapes, dtype=np.float64)
        ori_shapes = sw_shapes.copy()
        sw_shapes[:, 0] = sw_shapes[:, 0] * self.patch_w
        sw_shapes[:, 1] = sw_shapes[:, 1] * self.patch_h

        labels = data['labels'].copy()
        sw_labels_, ori_labels_, sw_shapes_, ori_shapes_, = [], [], [], []
        for i in range(self.num_imgs):
            ori_label = np.array(labels[i])
            sw_label = ori_label.copy()

            sw_label[:, -2] = sw_label[:, -2] / self.patch_w
            sw_label[:, -1] = sw_label[:, -1] / self.patch_h
            tiny_index = (sw_label[:, -2] <= MAX_W) * (sw_label[:, -1] <= MAX_H)
            sw_label_ = sw_label[tiny_index]
            if sw_label_.shape[0] > 0:
                sw_labels_.append(sw_label_)
                sw_shapes_.append(sw_shapes[i, :])

        self.labels = sw_labels_ + ori_labels_
        self.shapes = np.array(sw_shapes_ + ori_shapes_)

        if self.num_rate >= 1.:
            indices = np.arange(self.num_imgs).tolist() * self.num_rate
            random.shuffle(indices)
            self.indices = indices.copy()
            self.num_train = self.num_imgs * self.num_rate
        else:
            num_train = int(self.num_imgs * self.num_rate)
            indices = np.arange(self.num_imgs).tolist()
            random.shuffle(indices)
            self.indices = indices.copy()[:num_train]
            self.num_train = num_train

    def __len__(self):
        return self.num_train

    def get_data(self, idx, read_ori=False):
        hyp = self.hyp

        if read_ori:
            read_ori = (random.random() < self.ori_prob)
        """
        random cropping and check invalid annotations
        """
        if read_ori:  # read original image and annotation without cropping
            img, anno, anno_abs = self.get_ori_img(idx)
            if self.augment and random.random() < self.copy_paste_prob:  # random paste
                img, anno, anno_abs = self.copy_paste(img, anno, anno_abs)
            # if (img is not None) and (len(anno) > 0):
            #     img_ = img.copy()
            #     ih, iw = img_.shape[:2]
            #     for i in range(len(anno)):
            #         x, y, w, h = anno[i][1:]
            #         x, y, w, h = x * iw, y * ih, w * iw, h * ih
            #         x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
            #         box_ = [x1, y1, x2, y2]
            #         box_ = list(map(int, box_))
            #         cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
            #     cv2.imshow('ori', img_)
            anno, anno_abs = self.check_anno(anno, anno_abs, ori=True)
            # if (img is not None) and (len(anno) > 0):
            #     img_ = img.copy()
            #     ih, iw = img_.shape[:2]
            #     for i in range(len(anno)):
            #         x, y, w, h = anno[i][1:]
            #         x, y, w, h = x * iw, y * ih, w * iw, h * ih
            #         x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
            #         box_ = [x1, y1, x2, y2]
            #         box_ = list(map(int, box_))
            #         cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
            #     cv2.imshow('ori_checked', img_)
        else:
            if not self.augment:  # random cropping without augmentation
                # img, anno, anno_abs = self.fixed_crop(idx, self.patch_w, self.patch_h)
                img, anno, anno_abs = self.random_crop(idx, self.patch_w, self.patch_h, jitter=False)
            else:
                mosaic_prob = random.random()
                if mosaic_prob >= self.mosaic_prob_list[-1]:  # only random cropping
                    img, anno, anno_abs = self.random_crop(idx, self.patch_w, self.patch_h)
                else:  # crop over 2 images for mosaic image
                    delta = self.mosaic_prob_list - mosaic_prob
                    delta = (delta < 0.).astype(np.float32) * 100. + delta
                    func_index = np.argmin(delta)
                    img, anno, anno_abs = self.mosaic_func_list[func_index](idx)
            # if (img is not None) and (len(anno) > 0):
            #     img_ = img.copy()
            #     ih, iw = img_.shape[:2]
            #     for i in range(len(anno)):
            #         x, y, w, h = anno[i][1:]
            #         x, y, w, h = x * iw, y * ih, w * iw, h * ih
            #         x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
            #         box_ = [x1, y1, x2, y2]
            #         box_ = list(map(int, box_))
            #         cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
            #     cv2.imshow('crop', img_)
                if random.random() < self.copy_paste_prob:  # random paste
                    img, anno, anno_abs = self.copy_paste(img, anno, anno_abs)
            anno, anno_abs = self.check_anno(anno, anno_abs, ori=False)

        # img_ = img.copy()
        # ih, iw = img_.shape[:2]
        # for i in range(len(anno)):
        #     x, y, w, h = anno[i][1:]
        #     x, y, w, h = x * iw, y * ih, w * iw, h * ih
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # cv2.imshow('crop_checked', img_)
        # cv2.waitKey()

        """
        resize and pad the images to input size, and accordingly Convert the GT boxes
        """
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        img, ratio, pad = letterbox(img, self.img_size, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = np.array(anno)
        if labels.size:
            labels = labels.reshape((-1, 5))
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
        # img_ = img.copy()
        # for i in range(len(labels)):
        #     x1, y1, x2, y2 = labels[i, 1:]
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # cv2.imshow('ira', img_)
        # cv2.waitKey()

        """
        perspective augmentation
        """
        if self.augment and random.random() < self.perspective_prob:
            img, labels = random_perspective(img, labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
        # img_ = img.copy()
        # ih, iw = img_.shape[:2]
        # for i in range(len(labels)):
        #     x, y, w, h = labels[i, 1:]
        #     x, y, w, h = x * iw, y * ih, w * iw, h * ih
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # cv2.imshow('1', img_)

        """
        albumentation augmentation
        """
        if self.augment:
            # Albumentations
            if nl:
                img, labels = self.albumentations(img, labels)
            nl = len(labels)

            if random.random() < self.hsv_prob:
                augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        # img_ = img.copy()
        # ih, iw = img_.shape[:2]
        # for i in range(len(labels)):
        #     x, y, w, h = labels[i, 1:]
        #     x, y, w, h = x * iw, y * ih, w * iw, h * ih
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # cv2.imshow('2', img_)
        # cv2.waitKey()

        """
        Convert to tensor input
        """
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, str(self.sub / self.imgs[self.indices[idx]]), shapes, read_ori

    def __getitem__(self, idx):
        img, labels, path, shapes, read_ori = self.get_data(idx, read_ori=True)

        # img_ = img.detach().cpu().numpy()
        # img_ = np.ascontiguousarray(img_[::-1].transpose((1, 2, 0)))
        # ih, iw = img_.shape[:2]
        # for i in range(len(labels)):
        #     x, y, w, h = labels[i, 2:]
        #     x, y, w, h = x * iw, y * ih, w * iw, h * ih
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # cv2.imshow('0', img_)
        # cv2.waitKey()

        """
        only mixup when all the two images are cropped image
        """
        if not read_ori and self.augment and random.random() < self.hyp['mixup']:
            # img_ = img.detach().cpu().numpy()
            # img_ = np.ascontiguousarray(img_[::-1].transpose((1, 2, 0)))
            # ih, iw = img_.shape[:2]
            # for i in range(len(labels)):
            #     x, y, w, h = labels[i, 2:]
            #     x, y, w, h = x * iw, y * ih, w * iw, h * ih
            #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
            #     box_ = [x1, y1, x2, y2]
            #     box_ = list(map(int, box_))
            #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
            # cv2.imshow('0', img_)
            img2, labels2 = self.get_data(random.randint(0, self.num_train - 1), read_ori=read_ori)[:2]
            img, labels = mixup(img, labels, img2, labels2)
            # img_ = img2.detach().cpu().numpy()
            # img_ = np.ascontiguousarray(img_[::-1].transpose((1, 2, 0)))
            # ih, iw = img_.shape[:2]
            # for i in range(len(labels2)):
            #     x, y, w, h = labels2[i, 2:]
            #     x, y, w, h = x * iw, y * ih, w * iw, h * ih
            #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
            #     box_ = [x1, y1, x2, y2]
            #     box_ = list(map(int, box_))
            #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
            # cv2.imshow('1', img_)
            # img_ = img.detach().cpu().numpy()
            # img_ = np.ascontiguousarray(img_[::-1].transpose((1, 2, 0)))
            # ih, iw = img_.shape[:2]
            # for i in range(len(labels)):
            #     x, y, w, h = labels[i, 2:]
            #     x, y, w, h = x * iw, y * ih, w * iw, h * ih
            #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
            #     box_ = [x1, y1, x2, y2]
            #     box_ = list(map(int, box_))
            #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
            # cv2.imshow('2', img_)
            # cv2.waitKey()

        # img_ = img.detach().cpu().numpy()
        # img_ = np.ascontiguousarray(img_[::-1].transpose((1, 2, 0)))
        # ih, iw = img_.shape[:2]
        # for i in range(len(labels)):
        #     x, y, w, h = labels[i, 2:]
        #     x, y, w, h = x * iw, y * ih, w * iw, h * ih
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # cv2.imshow('0', img_)
        # cv2.waitKey()
        return img, labels, path, shapes

    def check_anno(self, anno, anno_abs, ori=False):
        new_anno = []
        new_anno_abs = []
        num = len(anno)
        if num > 0:
            for i in range(num):
                x, y, w, h = anno[i][1:]
                if (0. < x < 1.) and (0. < y < 1.):
                    if not ori:
                        if (0. < w <= MAX_W) and (0. < h <= MAX_H):
                            new_anno.append(anno[i])
                            new_anno_abs.append(anno_abs[i])
                    else:
                        if (MIN_W <= w < 1.) or (MIN_H <= h < 1.):
                            new_anno.append(anno[i])
                            new_anno_abs.append(anno_abs[i])
        return new_anno, new_anno_abs

    def get_ori_img(self, idx):
        index = self.indices[idx]
        img_name = self.imgs[index]
        anno = self.annos[index]
        anno_abs = self.annos_abs[index]
        img = read_img(str(self.sub / img_name), way=READ_WAY, read_bgr=True)
        if img is None:
            print('loss img: ' + img_name)
            return self.get_ori_img(random.randint(0, self.num_train - 1))
        else:
            return img, anno, anno_abs

    def fixed_crop(self, idx, patch_w_, patch_h_, jitter=True):
        index = self.indices[idx]
        img_name = self.imgs[index]
        anno = self.annos[index]
        anno_abs = self.annos_abs[index]

        img = read_img(str(self.sub / img_name), way=READ_WAY, read_bgr=True)
        ih, iw = img.shape[:2]

        patch_w, patch_h = patch_w_ * iw, patch_h_ * ih
        size_w, size_h = int(patch_w), int(patch_h)

        interval_w = max(math.ceil((self.num_patches_w * size_w - iw) // (self.num_patches_w - 1)), 0)
        interval_h = max(math.ceil((self.num_patches_h * size_h - ih) // (self.num_patches_h - 1)), 0)
        patches_x1 = np.arange(0, self.num_patches_w) * (size_w - interval_w)
        patches_y1 = np.arange(0, self.num_patches_h) * (size_h - interval_h)
        patches_x1[-1] = iw - size_w
        patches_y1[-1] = ih - size_h
        patches_x2, patches_y2 = patches_x1 + size_w, patches_y1 + size_h

        sub_imgs = []
        for j in range(self.num_patches_h):
            for i in range(self.num_patches_w):
                sub_img = img[patches_y1[j]: patches_y2[j], patches_x1[i]: patches_x2[i]]
                sub_imgs.append(sub_img)

        sub_annos, sub_annos_abs = [[] for i in range(self.num_patches)], [[] for i in range(self.num_patches)]
        if anno is None:
            index = random.randint(0, self.num_patches - 1)
            return sub_imgs[index], [], []

        else:
            num_objs = len(anno)
            for i in range(num_objs):
                cls = anno[i][0]
                x, y, w, h = anno_abs[i][1:]
                patch_xs = np.where((patches_x1 < x) & (patches_x2 > x))[0].tolist()
                patch_ys = np.where((patches_y1 < y) & (patches_y2 > y))[0].tolist()
                if len(patch_ys) and len(patch_xs):
                    for patch_x in patch_xs:
                        for patch_y in patch_ys:
                            x_, y_ = x - patches_x1[patch_x], y - patches_y1[patch_y]

                            x1, y1, x2, y2 = x_ - w // 2, y_ - h // 2, x_ + w // 2, y_ + h // 2
                            x1_, y1_, x2_, y2_ = max(0, x1), max(0, y1), min(size_w, x2), min(size_h, y2)

                            w, h, nw, nh = x2 - x1, y2 - y1, x2_ - x1_, y2_ - y1_
                            if (nw * nh > w * h * MAX_OCC) and min(nw / w, nh / h) > SHORT_THRESH:
                                nx, ny = int((x1_ + x2_) / 2), int((y1_ + y2_) / 2)

                                x_nor, y_nor, w_nor, h_nor = nx / size_w, ny / size_w, nw / size_w, ny / size_h
                                sub_annos[patch_y * self.num_patches_h + patch_x].append([cls, x_nor, y_nor, w_nor, h_nor])
                                sub_annos_abs[patch_y * self.num_patches_h + patch_x].append([cls, nx, ny, nw, nh])
                            else:
                                a = 1

        valid_indices = []
        for i in range(self.num_patches):
            if len(sub_annos[i]):
                valid_indices.append(i)

        if len(valid_indices):
            index = random.choice(valid_indices)
            new_img = sub_imgs[index]
            new_anno = sub_annos[index]
            new_anno_abs = sub_annos_abs[index]

            """DEBUG"""
            # # img_ = img.copy()
            # # for i in range(len(anno)):
            # #     x, y, w, h = anno[i][1:]
            # #     x, y, w, h = x * iw, y * ih, w * iw, h * ih
            # #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
            # #     box_ = [x1, y1, x2, y2]
            # #     box_ = list(map(int, box_))
            # #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
            # # cv2.imshow('0', img_)
            # crop_img_ = new_img.copy()
            # # for i in range(len(new_anno)):
            # #     x, y, w, h = new_anno[i][1:]
            # #     x, y, w, h = x * patch_w, y * patch_h, w * patch_w, h * patch_h
            # #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
            # #     box_ = [x1, y1, x2, y2]
            # #     box_ = list(map(int, box_))
            # #     cv2.rectangle(crop_img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
            # for i in range(len(new_anno_abs)):
            #     x, y, w, h = new_anno_abs[i][1:]
            #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
            #     box_ = [x1, y1, x2, y2]
            #     box_ = list(map(int, box_))
            #     cv2.rectangle(crop_img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
            # cv2.imshow('0', img)
            # cv2.imshow('1', crop_img_)
            # cv2.waitKey()
            return new_img, new_anno, new_anno_abs
        else:
            index = random.randint(0, self.num_patches - 1)
            return sub_imgs[index], [], []

    def get_jitter(self, jitter=None):
        scale_jitter = random.uniform(1., self.scale_jitter)
        if jitter is None:
            if random.random() > 0.5:
                scale_jitter = 1. / scale_jitter
        else:
            if jitter < 1.:
                scale_jitter = 1. / scale_jitter
        return scale_jitter

    def random_crop(self, idx, patch_w_, patch_h_, jitter=True):
        index = self.indices[idx]
        img_name = self.imgs[index]
        anno = self.annos[index]
        anno_abs = self.annos_abs[index]

        img = read_img(str(self.sub / img_name), way=READ_WAY, read_bgr=True)
        ih, iw = img.shape[:2]

        patch_w, patch_h = patch_w_ * iw, patch_h_ * ih
        if jitter and random.random() < self.scale_prob:
            if random.random() > 0.4:
                scale_jitter_w = self.get_jitter()
                scale_jitter_h = self.get_jitter(scale_jitter_w)
            else:
                scale_jitter = self.get_jitter()
                scale_jitter_w = scale_jitter_h = scale_jitter
            patch_w, patch_h = patch_w * scale_jitter_w, patch_h * scale_jitter_h
        patch_w, patch_h = min(iw, int(patch_w)), min(ih, int(patch_h))

        num_objs = len(anno)
        obj_index = random.randint(0, num_objs - 1) if num_objs > 1 else 0
        obj_anno_abs = anno_abs[obj_index]

        cls, x, y, w, h = obj_anno_abs
        x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        range_x = [max(0, int(x2 - patch_w)), min(iw - patch_w - 1, int(x1))]
        range_y = [max(0, int(y2 - patch_h)), min(ih - patch_h - 1, int(y1))]

        if not ((range_x[0] <= range_x[1]) and (range_y[0] <= range_y[1])):
            return self.random_crop(random.randint(0, self.num_train - 1), patch_w_, patch_h_, jitter=jitter)

        pos_x, pos_y = random.randint(range_x[0], range_x[1]), random.randint(range_y[0], range_y[1])
        crop_box = [pos_x, pos_y, pos_x + patch_w, pos_y + patch_h]
        crop_img = img[crop_box[1]: crop_box[3] + 1, crop_box[0]: crop_box[2] + 1]
        patch_h, patch_w = crop_img.shape[:2]

        new_anno, new_anno_abs = [], []
        for i in range(num_objs):
            cls, x, y, w, h = anno_abs[i]

            x, y = x - crop_box[0], y - crop_box[1]
            x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
            x1_, y1_, x2_, y2_ = max(0., x1), max(0., y1), min(patch_w - 1., x2), min(patch_h - 1., y2)
            w_, h_ = (x2_ - x1_), (y2_ - y1_)
            x_, y_ = (x1_ + x2_) / 2., (y1_ + y2_) / 2.

            if (w_ > 0.) and (h_ > 0.) and\
                    (w_ * h_ > w * h * self.max_occ) and min(w_ / w, h_ / h) > self.short_thresh:
                x_nor_, y_nor_, w_nor_, h_nor_ = x_ / patch_w, y_ / patch_h, w_ / patch_w, h_ / patch_h

                new_anno.append([cls, x_nor_, y_nor_, w_nor_, h_nor_])
                new_anno_abs.append([cls, x_, y_, w_, h_])

        # """DEBUG"""
        # # img_ = img.copy()
        # # for i in range(len(anno)):
        # #     x, y, w, h = anno[i][1:]
        # #     x, y, w, h = x * iw, y * ih, w * iw, h * ih
        # #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        # #     box_ = [x1, y1, x2, y2]
        # #     box_ = list(map(int, box_))
        # #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # # cv2.imshow('0', img_)
        # crop_img_ = crop_img.copy()
        # # for i in range(len(new_anno)):
        # #     x, y, w, h = new_anno[i][1:]
        # #     x, y, w, h = x * patch_w, y * patch_h, w * patch_w, h * patch_h
        # #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        # #     box_ = [x1, y1, x2, y2]
        # #     box_ = list(map(int, box_))
        # #     cv2.rectangle(crop_img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # for i in range(len(new_anno_abs)):
        #     x, y, w, h = new_anno_abs[i][1:]
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(crop_img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # cv2.imshow('1', crop_img_)
        # cv2.waitKey()
        return crop_img, new_anno, new_anno_abs

    def mosaic2_embed(self, idx):
        img0, _, anno0 = self.random_crop(idx, self.patch_w * random.uniform(self.embed_rate[0], self.embed_rate[1]),
                                          self.patch_h * random.uniform(self.embed_rate[0], self.embed_rate[1]))
        img1, _, anno1 = self.random_crop(random.randint(0, self.num_train - 1), self.patch_w, self.patch_h)

        ih0, iw0 = img0.shape[:2]
        ih1, iw1 = img1.shape[:2]
        if iw0 > iw1 * 0.8 and ih0 > ih1 * 0.8:
            return self.mosaic2_embed(random.randint(0, self.num_train - 1))

        xs, ys = np.arange(0, iw1 - iw0 - 1), np.arange(0, ih1 - ih0 - 1)
        if len(xs) <= 0 or len(ys) <= 0:
            return self.mosaic2_embed(random.randint(0, self.num_train - 1))
        pos_x, pos_y = random.choice(xs), random.choice(ys)
        embed_box = [pos_x, pos_y, pos_x + iw0, pos_y + ih0]

        new_img = img1.copy()
        new_img[pos_y: pos_y + ih0, pos_x: pos_x + iw0] = img0

        new_anno = []
        new_anno_abs = []
        for i in range(len(anno1)):
            cls, x, y, w, h = anno1[i]
            x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
            inter = [max(embed_box[0], x1), max(embed_box[1], y1), min(embed_box[2], x2), min(embed_box[3], y2)]
            inter_x, inter_y = max(inter[2] - inter[0], 0.), max(inter[3] - inter[1], 0.)
            if inter_x * inter_y < (1 - self.max_occ) / 2. * w * h:
                new_anno.append([cls, x / iw1, y / ih1, w / iw1, h / ih1])
                new_anno_abs.append([cls, x, y, w, h])

        for i in range(len(anno0)):
            cls, x, y, w, h = anno0[i]
            new_anno.append([cls, (x + pos_x) / iw1, (y + pos_y) / ih1, w / iw1, h / ih1])
            new_anno_abs.append([cls, x + pos_x, y + pos_y, w, h])

        # """DEBUG"""
        # img_ = new_img.copy()
        # # for i in range(len(new_anno)):
        # #     x, y, w, h = new_anno[i][1:]
        # #     x, y, w, h = x * nw, y * nh, w * nw, h * nh
        # #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        # #     box_ = [x1, y1, x2, y2]
        # #     box_ = list(map(int, box_))
        # #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # for i in range(len(new_anno_abs)):
        #     x, y, w, h = new_anno_abs[i][1:]
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # cv2.imshow('0', img_)
        # cv2.waitKey(10)
        # a = 1
        return new_img, new_anno, new_anno_abs

    def mosaic2_diag_r(self, idx):
        patch_w, patch_h = self.patch_w, self.patch_h
        img0, _, anno0 = self.random_crop(idx, patch_w, patch_h)
        img1, _, anno1 = self.random_crop(random.randint(0, self.num_train - 1), patch_w, patch_h)
        if random.random() > 0.5:
            img0, img1 = img1, img0
            anno0, anno1 = anno1, anno0

        ih0, iw0 = img0.shape[:2]
        ih1, iw1 = img1.shape[:2]
        nw, nh = max(iw0, iw1), max(ih0, ih1)
        nl = max(nw, nh)

        if iw1 > ih1:
            nw1 = nh1 = ih1
        else:
            nw1, nh1 = iw1, ih1
        if iw0 < ih0:
            nw0 = nh0 = iw0
        else:
            nw0, nh0 = iw0, ih0
        nw, nh = max(nw0, nw1), max(nh0, nh1)

        new_img0 = 114 * np.ones((nl, nl, 3), dtype=np.uint8)
        new_img1 = new_img0.copy()
        mask0 = np.triu(np.ones((nl, nl), dtype=np.uint8))[..., None]
        mask1 = (mask0 == 0.).astype(np.uint8)
        new_img0[:ih0, :iw0] = img0
        new_img1[:ih1, :iw1] = img1

        new_img_ = new_img0 * mask0 + new_img1 * mask1
        new_img = new_img_[:nh, :nw]

        new_anno = []
        new_anno_abs = []
        for i in range(len(anno0)):
            cls, x, y, w, h = anno0[i]
            if (x - w / 8.) - (y + h / 8.) > 1:
                new_anno.append([cls, x / nw, y / nh, w / nw, h / nh])
                new_anno_abs.append([cls, x, y, w, h])

        for i in range(len(anno1)):
            cls, x, y, w, h = anno1[i]
            if (y - h / 8.) - (x + w / 8.) > 1:
                new_anno.append([cls, x / nw, y / nh, w / nw, h / nh])
                new_anno_abs.append([cls, x, y, w, h])

        # """DEBUG"""
        # img_ = new_img.copy()
        # # for i in range(len(new_anno)):
        # #     x, y, w, h = new_anno[i][1:]
        # #     x, y, w, h = x * nw, y * nh, w * nw, h * nh
        # #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        # #     box_ = [x1, y1, x2, y2]
        # #     box_ = list(map(int, box_))
        # #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # for i in range(len(new_anno_abs)):
        #     x, y, w, h = new_anno_abs[i][1:]
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # cv2.imshow('0', img_)
        # cv2.imshow('1', new_img0)
        # cv2.imshow('2', new_img1)
        # cv2.imshow('3', new_img_)
        # cv2.waitKey()
        # a = 1
        return new_img, new_anno, new_anno_abs

    def mosaic2_diag_l(self, idx):
        patch_w, patch_h = self.patch_w, self.patch_h
        img0, _, anno0 = self.random_crop(idx, patch_w, patch_h)
        img1, _, anno1 = self.random_crop(random.randint(0, self.num_train - 1), patch_w, patch_h)

        ih0, iw0 = img0.shape[:2]
        ih1, iw1 = img1.shape[:2]
        nw, nh = max(iw0, iw1), max(ih0, ih1)
        nl = max(nw, nh)

        if iw1 > iw0 or ih1 > ih0:
            img0, img1 = img1, img0
            anno0, anno1 = anno1, anno0
            ih0, iw0 = img0.shape[:2]
            ih1, iw1 = img1.shape[:2]

        new_img0 = 114 * np.ones((nl, nl, 3), dtype=np.uint8)
        new_img1 = new_img0.copy()
        mask0 = np.triu(np.ones((nl, nl), dtype=np.uint8))[::-1][..., None]
        mask1 = (mask0 == 0.).astype(np.uint8)
        new_img0[:ih0, :iw0] = img0
        new_img1[:ih1, :iw1] = img1

        new_img_ = new_img0 * mask0 + new_img1 * mask1
        new_img = new_img_[:nh, :nw]

        new_anno = []
        new_anno_abs = []
        for i in range(len(anno0)):
            cls, x, y, w, h = anno0[i]
            if (x - w / 8.) + (y - h / 8.) > nl + 1:
                new_anno.append([cls, x / nw, y / nh, w / nw, h / nh])
                new_anno_abs.append([cls, x, y, w, h])

        for i in range(len(anno1)):
            cls, x, y, w, h = anno1[i]
            if (x + w / 8.) + (y + h / 8.) < nl - 1:
                new_anno.append([cls, x / nw, y / nh, w / nw, h / nh])
                new_anno_abs.append([cls, x, y, w, h])

        # """DEBUG"""
        # img_ = new_img.copy()
        # # for i in range(len(new_anno)):
        # #     x, y, w, h = new_anno[i][1:]
        # #     x, y, w, h = x * nw, y * nh, w * nw, h * nh
        # #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        # #     box_ = [x1, y1, x2, y2]
        # #     box_ = list(map(int, box_))
        # #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # for i in range(len(new_anno_abs)):
        #     x, y, w, h = new_anno_abs[i][1:]
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # cv2.imshow('0', img_)
        # cv2.imshow('1', new_img0)
        # cv2.imshow('2', new_img1)
        # cv2.imshow('3', new_img_)
        # cv2.waitKey()
        # a = 1
        return new_img, new_anno, new_anno_abs

    def mosaic2_horizontal(self, idx,
                           patch_w=None, patch_h=None,
                           imgs=None):
        if patch_w is None or patch_h is None:
            patch_w, patch_h = self.patch_w / 2., self.patch_h
        if imgs is None:
            img0, _, anno0 = self.random_crop(idx, patch_w, patch_h)
            img1, _, anno1 = self.random_crop(random.randint(0, self.num_train - 1), patch_w, patch_h)
        else:
            img0, anno0, img1, anno1 = imgs

        if random.random() > 0.5:
            img0, img1 = img1, img0
            anno0, anno1 = anno1, anno0

        ih0, iw0 = img0.shape[:2]
        ih1, iw1 = img1.shape[:2]
        nw, nh = iw0 + iw1, max(ih0, ih1)

        new_img = 114 * np.ones((nh, nw, 3), dtype=np.uint8)
        new_img[:ih0, :iw0] = img0
        new_img[:ih1, iw0:] = img1

        new_anno = []
        new_anno_abs = []
        for i in range(len(anno0)):
            cls, x, y, w, h = anno0[i]
            new_anno.append([cls, x / nw, y / nh, w / nw, h / nh])
            new_anno_abs.append([cls, x, y, w, h])

        for i in range(len(anno1)):
            cls, x, y, w, h = anno1[i]
            new_anno.append([cls, (x + iw0) / nw, y / nh, w / nw, h / nh])
            new_anno_abs.append([cls, x + iw0, y, w, h])

        # """DEBUG"""
        # img_ = new_img.copy()
        # # for i in range(len(new_anno)):
        # #     x, y, w, h = new_anno[i][1:]
        # #     x, y, w, h = x * nw, y * nh, w * nw, h * nh
        # #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        # #     box_ = [x1, y1, x2, y2]
        # #     box_ = list(map(int, box_))
        # #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # for i in range(len(new_anno_abs)):
        #     x, y, w, h = new_anno_abs[i][1:]
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # cv2.imshow('0', img_)
        # cv2.waitKey(10)
        # a = 1
        return new_img, new_anno, new_anno_abs

    def mosaic2_vertical(self, idx,
                         patch_w=None, patch_h=None,
                         imgs=None):
        if patch_w is None or patch_h is None:
            patch_w, patch_h = self.patch_w, self.patch_h / 2.
        if imgs is None:
            img0, _, anno0 = self.random_crop(idx, patch_w, patch_h)
            img1, _, anno1 = self.random_crop(random.randint(0, self.num_train - 1), patch_w, patch_h)
        else:
            img0, anno0, img1, anno1 = imgs

        if random.random() > 0.5:
            img0, img1 = img1, img0
            anno0, anno1 = anno1, anno0

        ih0, iw0 = img0.shape[:2]
        ih1, iw1 = img1.shape[:2]
        nw, nh = max(iw0, iw1), ih0 + ih1

        new_img = 114 * np.ones((nh, nw, 3), dtype=np.uint8)
        new_img[:ih0, :iw0] = img0
        new_img[ih0:, :iw1] = img1

        new_anno = []
        new_anno_abs = []
        for i in range(len(anno0)):
            cls, x, y, w, h = anno0[i]
            new_anno.append([cls, x / nw, y / nh, w / nw, h / nh])
            new_anno_abs.append([cls, x, y, w, h])

        for i in range(len(anno1)):
            cls, x, y, w, h = anno1[i]
            new_anno.append([cls, x / nw, (y + ih0) / nh, w / nw, h / nh])
            new_anno_abs.append([cls, x, y + ih0, w, h])

        # """DEBUG"""
        # img_ = new_img.copy()
        # # for i in range(len(new_anno)):
        # #     x, y, w, h = new_anno[i][1:]
        # #     x, y, w, h = x * nw, y * nh, w * nw, h * nh
        # #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        # #     box_ = [x1, y1, x2, y2]
        # #     box_ = list(map(int, box_))
        # #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # for i in range(len(new_anno_abs)):
        #     x, y, w, h = new_anno_abs[i][1:]
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # cv2.imshow('0', img_)
        # cv2.waitKey(10)
        # a = 1
        return new_img, new_anno, new_anno_abs

    def mosaic3(self, idx):
        if random.random() > 0.5:
            img0, _, anno0 = self.random_crop(idx, patch_w_=self.patch_w/2., patch_h_=self.patch_h)
            img1, _, anno1 = self.mosaic2_vertical(random.randint(0, self.num_train - 1),
                                                   patch_w=self.patch_w/2., patch_h=self.patch_h/2.)
            img, anno, anno_abs = self.mosaic2_horizontal(idx=0, imgs=[img0, anno0, img1, anno1])
        else:
            img0, _, anno0 = self.random_crop(idx, patch_w_=self.patch_w, patch_h_=self.patch_h/2.)
            img1, _, anno1 = self.mosaic2_horizontal(random.randint(0, self.num_train - 1),
                                                     patch_w=self.patch_w/2., patch_h=self.patch_h/2.)
            img, anno, anno_abs = self.mosaic2_vertical(idx=0, imgs=[img0, anno0, img1, anno1])

        # """DEBUG"""
        # img_ = img.copy()
        # for i in range(len(anno_abs)):
        #     x, y, w, h = anno_abs[i][1:]
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # cv2.imshow('0', img_)
        # cv2.waitKey(10)
        # a = 1
        return img, anno, anno_abs

    def mosaic4(self, idx):
        patch_w, patch_h = self.patch_w / 2., self.patch_h / 2.

        imgs, annos = [], []
        for i in range(4):
            img, _, anno = self.random_crop(idx, patch_w, patch_h)
            imgs.append(img), annos.append(anno)
            idx = random.randint(0, self.num_train - 1)

        indices = np.arange(0, 4).tolist()
        random.shuffle(indices)
        img0, anno0 = imgs[indices[0]], annos[indices[0]]
        img1, anno1 = imgs[indices[1]], annos[indices[1]]
        img2, anno2 = imgs[indices[2]], annos[indices[2]]
        img3, anno3 = imgs[indices[3]], annos[indices[3]]

        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]
        crop_w_left, crop_w_right = min(w0, w2), min(w1, w3)
        crop_h_up, crop_h_down = min(h0, h1), min(h2, h3)

        img0 = img0[(h0 - crop_h_up):, (w0 - crop_w_left):]
        img1 = img1[(h1 - crop_h_up):, :crop_w_right]
        img2 = img2[:crop_h_down, (w2 - crop_w_left):]
        img3 = img3[:crop_h_down, :crop_w_right]

        ih0, iw0 = img0.shape[:2]
        ih1, iw1 = img1.shape[:2]
        ih2, iw2 = img2.shape[:2]
        ih3, iw3 = img3.shape[:2]
        nw, nh = iw0 + iw1, ih0 + ih2
        new_img = np.zeros((nh, nw, 3), dtype=np.uint8)
        new_img[:ih0, :iw0] = img0
        new_img[:ih0, iw0:] = img1
        new_img[ih0:, :iw0] = img2
        new_img[ih0:, iw0:] = img3

        new_anno = []
        new_anno_abs = []
        for i in range(len(anno0)):
            anno = anno0[i]
            cls, x, y, w, h = anno[0], anno[1] - (w0 - crop_w_left), anno[2] - (h0 - crop_h_up), anno[3], anno[4]

            x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
            x1_, y1_, x2_, y2_ = max(0., x1), max(0., y1), min(iw0 - 1., x2), min(ih0 - 1., y2)
            w_, h_ = (x2_ - x1_), (y2_ - y1_)
            if (w_ > 0.) and (h_ > 0.) and (w_ * h_ > w * h * self.max_occ) and min(w_ / w, h_ / h) > self.short_thresh:
                x_, y_ = (x1_ + x2_) / 2., (y1_ + y2_) / 2.
                new_anno.append([cls, x_ / nw, y_ / nh, w_ / nw, h_ / nh])
                new_anno_abs.append([cls, x_, y_, w_, h_])

        for i in range(len(anno1)):
            anno = anno1[i]
            cls, x, y, w, h = anno[0], anno[1], anno[2] - (h1 - crop_h_up), anno[3], anno[4]

            x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
            x1_, y1_, x2_, y2_ = max(0., x1), max(0., y1), min(iw1 - 1., x2), min(ih1 - 1., y2)
            w_, h_ = (x2_ - x1_), (y2_ - y1_)
            if (w_ > 0.) and (h_ > 0.) and (w_ * h_ > w * h * self.max_occ) and min(w_ / w, h_ / h) > self.short_thresh:
                x_, y_ = (x1_ + x2_) / 2., (y1_ + y2_) / 2.
                new_anno.append([cls, (x_ + iw0) / nw, y_ / nh, w_ / nw, h_ / nh])
                new_anno_abs.append([cls, x_ + iw0, y_, w_, h_])

        for i in range(len(anno2)):
            anno = anno2[i]
            cls, x, y, w, h = anno[0], anno[1] - (w2 - crop_w_left), anno[2], anno[3], anno[4]

            x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
            x1_, y1_, x2_, y2_ = max(0., x1), max(0., y1), min(iw2 - 1., x2), min(ih2 - 1., y2)
            w_, h_ = (x2_ - x1_), (y2_ - y1_)
            if (w_ > 0.) and (h_ > 0.) and (w_ * h_ > w * h * self.max_occ) and min(w_ / w, h_ / h) > self.short_thresh:
                x_, y_ = (x1_ + x2_) / 2., (y1_ + y2_) / 2.
                new_anno.append([cls, x_ / nw, (y_ + ih0) / nh, w_ / nw, h_ / nh])
                new_anno_abs.append([cls, x_, y_ + ih0, w_, h_])

        for i in range(len(anno3)):
            anno = anno3[i]
            cls, x, y, w, h = anno[0], anno[1], anno[2], anno[3], anno[4]

            x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
            x1_, y1_, x2_, y2_ = max(0., x1), max(0., y1), min(iw3 - 1., x2), min(ih3 - 1., y2)
            w_, h_ = (x2_ - x1_), (y2_ - y1_)
            if (w_ > 0.) and (h_ > 0.) and (w_ * h_ > w * h * self.max_occ) and min(w_ / w, h_ / h) > self.short_thresh:
                x_, y_ = (x1_ + x2_) / 2., (y1_ + y2_) / 2.
                new_anno.append([cls, (x_ + iw0) / nw, (y_ + ih0) / nh, w_ / nw, h_ / nh])
                new_anno_abs.append([cls, x_ + iw0, y_ + ih0, w_, h_])

        # """DEBUG"""
        # img_ = new_img.copy()
        # # for i in range(len(new_anno)):
        # #     x, y, w, h = new_anno[i][1:]
        # #     x, y, w, h = x * nw, y * nh, w * nw, h * nh
        # #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        # #     box_ = [x1, y1, x2, y2]
        # #     box_ = list(map(int, box_))
        # #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # for i in range(len(new_anno_abs)):
        #     x, y, w, h = new_anno_abs[i][1:]
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # cv2.imshow('0', img_)
        # cv2.waitKey(10)
        # a = 1
        return new_img, new_anno, new_anno_abs


    def copy_paste(self, img, anno, anno_abs):
        MAX_OBJ_NUM = 80
        num_objs = len(anno)
        if num_objs == 0:
            if random.random() > 0.5:
                img, anno, anno_abs = self.empty_paste(img)
            else:
                return img, anno, anno_abs
        elif num_objs > MAX_OBJ_NUM:
            return img, anno, anno_abs

        # 随意粘贴与相似粘贴/聚类粘贴分别独立进行
        paste_prob1 = random.random()
        if paste_prob1 < self.easy_paste_prob:
            img, anno, anno_abs = self.easy_paste(img, anno, anno_abs)

        if len(anno) < MAX_OBJ_NUM:
            paste_prob2 = random.random()
            if paste_prob2 < self.similar_paste_prob:
                img, anno, anno_abs = self.similar_paste(img, anno, anno_abs)
            elif paste_prob2 < self.cluster_paste_prob:
                img, anno, anno_abs = self.cluster_paste(img, anno, anno_abs)

            # # 没有任何一种粘贴实现时, 强制进行随意粘贴
            # elif paste_prob1 >= self.easy_paste_prob:
            #     img, anno, anno_abs = self.easy_paste(img, anno, anno_abs)
        return img, anno, anno_abs

    def empty_paste(self, img):
        ih, iw = img.shape[:2]
        num_objs = random.randint(1, 2)  # 从其他图像中随机选择的obj数量

        crop_range = (0.9, 1.15)
        crop_num_rate = 0.5
        dist_range = (0.85, 1.15)
        dist_num_rate = 0.3  # 30%的obj进行长宽比例不一致的缩放，从而改变长宽比
        scale_range = 0.25  # 缩放倍数: 0.25——4.0
        min_size = 10  # 尺寸条件：粘贴的obj的边长不能小于该值
        max_overlap = 0.10  # obj与原目标以及obj之间的覆盖率阈值
        num_try = 200

        objs = []
        for i in range(num_objs):
            cls = random.choice(self.valid_cls)  # 所有类别具有相同的概率，可进一步修改，以方便进行重采样
            obj = random.choice(self.objs[str(cls)])
            objs.append([cls] + obj)
        objs = np.array(objs).astype(np.int64)
        all_wh = objs[:, -2:]

        all_whs0 = np.tile(all_wh[:, None, :], (1, num_try, 1))
        # 第二次随机尺寸变换：随机挑选部分obj进行随机的畸变
        dist_num = int(num_try * dist_num_rate)
        dist_matrix = np.random.uniform(dist_range[0], dist_range[-1], (dist_num, num_objs, 2))
        dist_matrix = np.concatenate([dist_matrix, np.ones((num_try - dist_num, num_objs, 2))], axis=0)
        np.random.shuffle(dist_matrix)
        dist_matrix = dist_matrix.transpose((1, 0, 2))
        all_whs1 = (all_whs0 * dist_matrix).astype(np.int64)
        # 第三次随机尺寸变换：随机挑选部分obj, 随机地进行整体放/缩
        scale_factor0 = np.ones((num_objs, num_try // 2, 1))
        scale_factor1 = np.random.uniform(scale_range, 1., (num_objs, num_try // 4, 1))
        scale_factor2 = np.random.uniform(1., 1. / scale_range, (num_objs, num_try // 4, 1))
        scale_factor = np.concatenate([scale_factor0, scale_factor1, scale_factor2], axis=1)
        all_whs2 = (all_whs1 * scale_factor).astype(np.int64)

        all_xys = np.random.random((num_objs, num_try, 2)) * np.array([iw, ih]).reshape((1, 1, 2))
        all_xywh = np.concatenate((all_xys, all_whs2), axis=-1).astype(np.int64)
        all_xyxy = xywh2xyxy_(all_xywh)
        clip_coords_(all_xyxy, (ih, iw))

        # 尺寸条件1：粘贴后的物体不能超出边界
        all_xywh_ = xyxy2xywh_(all_xyxy)
        all_whs_ = all_xywh_[..., 2:]
        size_mask0 = (all_whs_ / np.maximum(all_whs2, 1e-6)) == 1.
        # 尺寸条件2：裁剪物体时，绝对尺寸不得小于8×8
        size_mask1 = all_whs_ >= min_size
        size_mask = size_mask0 * size_mask1
        size_mask = size_mask[..., 0] * size_mask[..., 1]
        num_proper = size_mask.astype(np.int64).sum(1)

        if num_proper.sum() == 0:
            return self.empty_paste(img)

        paste_objs = []
        indices = []
        for i in range(num_objs):
            index = np.where(size_mask[i])
            index = np.random.choice(index[0])
            indices.append([i, index])
            if len(paste_objs):
                paste_objs = np.concatenate([paste_objs, all_xyxy[i][index: index+1]])
            else:
                paste_objs = all_xyxy[i][index: index+1]

        # 位置条件2：粘贴后的物体之间也不能互相遮盖
        overlap1, overlap2 = boxes_iou(paste_objs[:, None, :], paste_objs[None, ...], split=True)[1:]
        num_chosens = len(paste_objs)
        mask = np.eye(num_chosens, dtype=np.bool)
        overlap1[mask], overlap2[mask] = 0., 0.
        mask = (overlap1 > max_overlap) + (overlap2 > max_overlap)
        mask = mask.astype(np.int64)

        # 将不满足位置条件2的objs删除
        del_objs = []
        while np.any(mask):
            over_num = mask.sum(0)
            max_over = over_num.argmax()
            mask[max_over, :], mask[:, max_over] = 0, 0
            del_objs.append(max_over)
        # 无符合条件的位置，结束copy-paste
        if len(del_objs) == num_chosens:
            return self.empty_paste(img)
        if len(del_objs):
            paste_objs = np.delete(paste_objs, del_objs, axis=0)
            indices = [item for i, item in enumerate(indices) if i not in del_objs]

        paste_objs_xywh = xyxy2xywh_(paste_objs)
        # 将objs粘贴, 过程包括: 选择并读取图像, 根据wh1进行crop, 根据wh3进行缩放，根据paste objs粘贴
        paste_img = img.copy()
        paste_anno, paste_anno_abs = [], []
        img_fore, img_id_fore = None, -2
        num_chosens = len(paste_objs)
        for i in range(num_chosens):
            # 选择并读取图像
            obj_id, obj_num = indices[i]
            cls = objs[obj_id][0]
            crop_xywh = objs[obj_id, 2:]
            img_id = int(objs[obj_id, 1])
            if img_id == img_id_fore:
                ori_img = img_fore.copy()
            else:
                if img_id == -1:
                    ori_img = img.copy()
                else:
                    ori_img = read_img(str(self.sub / self.obj_imgs[img_id]), way=READ_WAY, read_bgr=True)
            img_fore = ori_img.copy()
            img_id_fore = img_id

            scale_wh = all_whs2[obj_id, obj_num]
            paste_pos = paste_objs[i]

            # 建立crop box并在图像上进行crop
            crop_xyxy = xywh2xyxy_(crop_xywh)
            clip_coords_(crop_xyxy, ori_img.shape)
            crop_xyxy = crop_xyxy.astype(np.int64)
            crop_obj = ori_img[crop_xyxy[1]: crop_xyxy[3], crop_xyxy[0]: crop_xyxy[2]]
            # 放缩
            scale_obj = cv2.resize(crop_obj, (scale_wh[0], scale_wh[1]))
            # 粘贴
            paste_img[paste_pos[1]: paste_pos[3], paste_pos[0]: paste_pos[2], :] = scale_obj

            x, y, w, h = paste_objs_xywh[i]
            nx, ny, nw, nh = x / iw, y / ih, w / iw, h / ih
            paste_anno.append([cls, nx, ny, nw, nh])
            paste_anno_abs.append([cls, x, y, w, h])

        # img_ = paste_img.copy()
        # if paste_anno_abs:
        #     for j in range(len(paste_anno_abs)):
        #         x, y, w, h = paste_anno_abs[j][1:]
        #         x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #         box_ = [x1, y1, x2, y2]
        #         box_ = list(map(int, box_))
        #         cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 0, 255), 2)
        # cv2.imshow('2', img_)
        # cv2.waitKey()
        return paste_img, paste_anno, paste_anno_abs

    # 完全随机放置，数量少的类别有更大几率被选取，相当于进行re-sampling，缓解类别不均衡的长尾问题
    def easy_paste(self, img, anno, anno_abs, supplement=False):
        ih, iw = img.shape[:2]
        num_objs = len(anno_abs)
        objs = np.array(anno_abs).reshape((-1, 5))
        objs_xywh = objs[..., 1:]
        objs_xyxy = xywh2xyxy_(objs_xywh)

        # 剔除特殊类别
        if len(self.ban_cls):
            objs_cls = objs[:, 0].astype(np.int64)
            indices = []
            for i in range(num_objs):
                if objs_cls[i] not in self.ban_cls:
                    indices.append(i)
            num_objs = len(indices)
            if num_objs > 0:
                objs = objs[indices]

        crop_range = (0.9, 1.15)
        crop_num_rate = 0.5
        dist_range = (0.9, 1.1)
        dist_num_rate = 0.3  # 30%的obj进行长宽比例不一致的缩放，从而改变长宽比
        scale_range = 0.5  # 缩放倍数: 0.5——2.0
        min_size = 24  # 尺寸条件：粘贴的obj的边长不能小于该值
        max_overlap = 0.12  # obj与原目标以及obj之间的覆盖率阈值

        num_try = 200  # 每个obj随机生成200个粘贴位置
        if supplement:
            max_chosen = 1  # 每个obj最多粘贴几次
            num_self = 0
            num_other = random.randint(1, 2)  # 从其他图像中随机选择的obj数量
        else:
            max_chosen = 2  # 每个obj最多粘贴几次
            num_self = random.randint(0, min(3, num_objs))  # 当前图像中随机选择的obj数量
            num_other = random.randint(1, 4)  # 从其他图像中随机选择的obj数量

        # 随机地从其他图像中选择粘贴物体
        other_objs = []
        while len(other_objs) < num_other:
            cls = random.choice(self.valid_cls)  # 所有类别具有相同的概率，可进一步修改，以方便进行重采样
            other_obj = random.choice(self.objs[str(cls)])
            other_objs.append([cls] + other_obj)
        other_objs = np.array(other_objs)
        other_objs = other_objs[other_objs[:, 1].argsort()]

        # 随机地从当前图像中选择粘贴物体
        self_objs = []
        if num_self:
            for i in range(num_self):
                self_index = random.randint(0, num_objs - 1)
                self_obj = objs[self_index]
                self_obj = [self_obj[0], -1] + self_obj[1:].tolist()
                self_objs.append(self_obj)
            self_objs = np.array(self_objs)

            all_objs = np.concatenate([other_objs, self_objs], axis=0)
        else:
            all_objs = other_objs

        all_wh = all_objs[:, -2:]
        num_all = num_other + num_self

        # img_fore, img_id_fore = None, -2
        # for i in range(num_all):
        #     # 在当前图像上选择obj
        #     if i >= num_other:
        #         ori_img = img.copy()
        #         cls = self_objs[i - num_other, 0]
        #         crop_xy = self_xywh[i - num_other, :2]
        #     # 从其他图像上选择obj
        #     else:
        #         cls, img_id = other_objs[i, :2]
        #         crop_xy = other_xywh[i, :2]
        #
        #         if img_id == img_id_fore:
        #             ori_img = img_fore.copy()
        #         else:
        #             ori_img = read_img(str(self.sub / self.obj_imgs[img_id]), way=READ_WAY, read_bgr=True)
        #
        #         img_fore = ori_img.copy()
        #         img_id_fore = img_id
        #     crop_wh = all_wh[i]
        #
        #     crop_xywh = np.concatenate([crop_xy, crop_wh]).reshape((-1, 4)).astype(np.int64)
        #     crop_xyxy = xywh2xyxy_(crop_xywh)
        #     clip_coords_(crop_xyxy, ori_img.shape)
        #     crop_xyxy = crop_xyxy[0]
        #     crop_obj = ori_img[crop_xyxy[1]: crop_xyxy[3], crop_xyxy[0]: crop_xyxy[2]]
        #
        #     crop_obj_ = cv2.resize(crop_obj, (10 * (crop_xyxy[2] - crop_xyxy[0]), 10 * (crop_xyxy[3] - crop_xyxy[1])))
        #     cv2.imshow('1', crop_obj)
        #     cv2.imshow('2', crop_obj_)
        #     cv2.waitKey()

        all_whs0 = np.tile(all_wh[:, None, :], (1, num_try, 1))
        # # 第一次随机尺寸变换：随机挑选一部分obj, 使裁剪时的边界随机扩张或收缩
        # crop_num = int(num_try * 0.5)
        # crop_matrix = np.random.uniform(crop_range[0], crop_range[-1], (crop_num, num_all, 2))
        # crop_matrix = np.concatenate([crop_matrix, np.ones((num_try - crop_num, num_all, 2))], axis=0)
        # np.random.shuffle(crop_matrix)
        # crop_matrix = crop_matrix.transpose((1, 0, 2))
        # all_whs1 = (all_whs0 * crop_matrix).astype(np.int64)
        all_whs1 = all_whs0.astype(np.int64)
        # 第二次随机尺寸变换：随机挑选部分obj进行随机的畸变
        dist_num = int(num_try * dist_num_rate)
        dist_matrix = np.random.uniform(dist_range[0], dist_range[-1], (dist_num, num_all, 2))
        dist_matrix = np.concatenate([dist_matrix, np.ones((num_try - dist_num, num_all, 2))], axis=0)
        np.random.shuffle(dist_matrix)
        dist_matrix = dist_matrix.transpose((1, 0, 2))
        all_whs2 = (all_whs1 * dist_matrix).astype(np.int64)
        # 第三次随机尺寸变换：随机挑选部分obj, 随机地进行整体放/缩
        scale_factor0 = np.ones((num_all, num_try // 2, 1))
        scale_factor1 = np.random.uniform(scale_range, 1., (num_all, num_try // 4, 1))
        scale_factor2 = np.random.uniform(1., 1. / scale_range, (num_all, num_try // 4, 1))
        scale_factor = np.concatenate([scale_factor0, scale_factor1, scale_factor2], axis=1)
        all_whs3 = (all_whs2 * scale_factor).astype(np.int64)

        all_xys = np.random.random((num_all, num_try, 2)) * np.array([iw, ih]).reshape((1, 1, 2))
        all_xywh = np.concatenate((all_xys, all_whs3), axis=-1).astype(np.int64)
        all_xyxy = xywh2xyxy_(all_xywh)
        clip_coords_(all_xyxy, (ih, iw))

        # 尺寸条件1：粘贴后的物体不能超出边界
        all_xywh_ = xyxy2xywh_(all_xyxy)
        all_whs_ = all_xywh_[..., 2:]
        size_mask0 = (all_whs_ / np.maximum(all_whs3, 1e-6)) == 1.
        # 尺寸条件2：裁剪物体时，绝对尺寸不得小于8×8
        size_mask1 = all_whs_ >= min_size
        size_mask = size_mask0 * size_mask1
        size_mask = size_mask[..., 0] * size_mask[..., 1]

        # 位置条件1：粘贴后的物体不能遮盖原来的目标
        overlap1, overlap2 = boxes_iou(all_xyxy.reshape((-1, 1, 4)), objs_xyxy[None, ...], split=True)[1:]
        overlap1, overlap2 = overlap1.reshape((num_all, num_try, -1)), overlap2.reshape((num_all, num_try, -1))
        overlap1_max, overlap2_max = overlap1.max(-1), overlap2.max(-1)
        overlap_mask = (overlap1_max <= max_overlap) * (overlap2_max <= max_overlap)
        mask = overlap_mask * size_mask
        num_proper = mask.astype(np.int64).sum(1)

        # 无符合条件的位置，结束copy-paste
        if num_proper.sum() == 0:
            return img, anno, anno_abs

        # 随机挑选出符合尺寸条件与位置条件的box
        num_chosen = np.random.randint(1, max_chosen+1, num_all)
        num_chosen = np.minimum(num_proper, num_chosen)

        paste_objs = []
        indices = []
        for i in range(num_all):
            if num_chosen[i] > 0:
                index = np.where(mask[i])
                index = np.random.choice(index[0], num_chosen[i])
                for j in index:
                    indices.append([i, j])
                if len(paste_objs):
                    paste_objs = np.concatenate([paste_objs, all_xyxy[i][index]])
                else:
                    paste_objs = all_xyxy[i][index]

        # 位置条件2：粘贴后的物体之间也不能互相遮盖
        overlap1, overlap2 = boxes_iou(paste_objs[:, None, :], paste_objs[None, ...], split=True)[1:]
        num_chosens = len(paste_objs)
        mask = np.eye(num_chosens, dtype=np.bool)
        overlap1[mask], overlap2[mask] = 0., 0.
        mask = (overlap1 > max_overlap) + (overlap2 > max_overlap)
        mask = mask.astype(np.int64)

        # 将不满足位置条件2的objs删除
        del_objs = []
        while np.any(mask):
            over_num = mask.sum(0)
            max_over = over_num.argmax()
            mask[max_over, :], mask[:, max_over] = 0, 0
            del_objs.append(max_over)
        # 无符合条件的位置，结束copy-paste
        if len(del_objs) == num_chosens:
            return img, anno, anno_abs
        if len(del_objs):
            paste_objs = np.delete(paste_objs, del_objs, axis=0)
            indices = [item for i, item in enumerate(indices) if i not in del_objs]

        paste_objs_xywh = xyxy2xywh_(paste_objs)
        # 将objs粘贴, 过程包括: 选择并读取图像, 根据wh1进行crop, 根据wh3进行缩放，根据paste objs粘贴
        paste_img = img.copy()
        paste_anno, paste_anno_abs = [], []
        img_fore, img_id_fore = None, -2
        num_chosens = len(paste_objs)
        for i in range(num_chosens):
            # 选择并读取图像
            obj_id, obj_num = indices[i]
            cls = all_objs[obj_id][0]
            crop_xywh = all_objs[obj_id, 2:]
            img_id = int(all_objs[obj_id, 1])
            if img_id == img_id_fore:
                ori_img = img_fore.copy()
            else:
                if img_id == -1:
                    ori_img = img.copy()
                else:
                    ori_img = read_img(str(self.sub / self.obj_imgs[img_id]), way=READ_WAY, read_bgr=True)
            img_fore = ori_img.copy()
            img_id_fore = img_id

            scale_wh = all_whs3[obj_id, obj_num]
            paste_pos = paste_objs[i]

            # 建立crop box并在图像上进行crop
            crop_xyxy = xywh2xyxy_(crop_xywh)
            clip_coords_(crop_xyxy, ori_img.shape)
            crop_xyxy = crop_xyxy.astype(np.int64)
            crop_obj = ori_img[crop_xyxy[1]: crop_xyxy[3], crop_xyxy[0]: crop_xyxy[2]]
            # 放缩
            scale_obj = cv2.resize(crop_obj, (scale_wh[0], scale_wh[1]))
            # 粘贴
            paste_img[paste_pos[1]: paste_pos[3], paste_pos[0]: paste_pos[2], :] = scale_obj

            x, y, w, h = paste_objs_xywh[i]
            nx, ny, nw, nh = x / iw, y / ih, w / iw, h / ih
            paste_anno.append([cls, nx, ny, nw, nh])
            paste_anno_abs.append([cls, x, y, w, h])

        # img_ = paste_img.copy()
        # for j in range(num_objs):
        #     x, y, w, h = anno_abs[j][1:]
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # if paste_anno_abs:
        #     for j in range(len(paste_anno_abs)):
        #         x, y, w, h = paste_anno_abs[j][1:]
        #         x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #         box_ = [x1, y1, x2, y2]
        #         box_ = list(map(int, box_))
        #         cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 0, 255), 2)
        # cv2.imshow('2', img_)
        # cv2.waitKey()

        # 不要忘了将objs加入anno和anno abs中
        anno = anno + paste_anno
        anno_abs = anno_abs + paste_anno_abs
        return paste_img, anno, anno_abs

    # 先随机选择一个物体，再随机选择同类物体，放置在其附近
    def similar_paste(self, img, anno, anno_abs):
        ih, iw = img.shape[:2]
        num_objs = len(anno_abs)
        objs = np.array(anno_abs).reshape((-1, 5))
        objs_xywh = objs[..., 1:]
        objs_xyxy = xywh2xyxy_(objs_xywh)

        # 剔除特殊类别
        if len(self.ban_cls):
            objs_cls = objs[:, 0].astype(np.int64)
            indices = []
            for i in range(num_objs):
                if objs_cls[i] not in self.ban_cls:
                    indices.append(i)
            num_objs = len(indices)
            if num_objs > 0:
                objs = objs[indices]
                objs_xywh = objs_xywh[indices]
            else:
                while num_objs == 0:
                    img, anno, anno_abs = self.easy_paste(img, anno, anno_abs, supplement=True)
                    num_objs = len(anno_abs)
                    objs = np.array(anno_abs).reshape((-1, 5))
                    objs_xywh = objs[..., 1:]
                    objs_xyxy = xywh2xyxy_(objs_xywh)
                    objs_cls = objs[:, 0].astype(np.int64)
                    indices = []
                    for i in range(num_objs):
                        if objs_cls[i] not in self.ban_cls:
                            indices.append(i)
                    num_objs = len(indices)
                objs = objs[indices]
                objs_xywh = objs_xywh[indices]

        crop_range = (0.9, 1.15)
        crop_num_rate = 0.5
        dist_range = (0.9, 1.1)
        dist_num_rate = 0.3  # 30%的obj进行长宽比例不一致的缩放，从而改变长宽比
        scale_range = 0.5  # 缩放倍数: 0.5——2.0
        min_size = 24  # 尺寸条件：粘贴的obj的边长不能小于该值
        max_overlap = 0.12  # obj与原目标以及obj之间的覆盖率阈值
        paste_r = 1.5  # 粘贴位置半径

        num_try = 200  # 每个obj随机生成200个粘贴位置
        max_chosen = 2  # 每个obj最多粘贴几次
        num_self = random.randint(1, min(3, num_objs))  # 粘贴种子数
        num_others = np.random.randint(1, 4, num_self)  # 每个粘贴种子附近的obj数(种子本身也在粘贴队列中)

        """
        1.先随机挑选几个目标作为种子/锚(粘贴围绕的中心)
        2.依次对每个种子随机选择同类别的粘贴物体，并且自身也加入粘贴队列中
        3.筛选得到每个粘贴中心周围的可粘贴位置(需要将粘贴物体的尺寸放缩到与原目标差不多)
        4.裁剪、缩放、粘贴，并加入标注
        """
        # 随机地从当前图像中选择粘贴中心（锚/种子）
        self_index = [random.randint(0, num_objs - 1) for i in range(num_self)]
        self_objs = objs[self_index]
        self_clses = self_objs[:, 0]
        self_xywhs = objs_xywh[self_index]
        self_whs = self_xywhs[..., 2:]

        paste_anno, paste_anno_abs = [], []
        paste_img = img.copy()
        # 随机地从其他图像中选择粘贴物体, 同时记录原位置尺寸(裁剪时需要坐标) 以及 重塑后的位置尺寸(粘贴时需要的尺寸)
        for i in range(num_self):
            obj_cls = self_clses[i]
            self_wh = self_whs[i]
            num_other = num_others[i]
            all_other_objs = self.objs[str(int(obj_cls))]
            num_other_objs = self.obj_num[str(int(obj_cls))]

            # 随机地从其他图像中选择粘贴物体, 记录原尺寸坐标
            other_objs = []
            for j in range(num_other):
                other_obj = all_other_objs[random.randint(0, num_other_objs - 1)]
                other_objs.append([i, obj_cls] + other_obj)
            other_objs = np.array(other_objs)
            other_objs = other_objs[other_objs[:, 2].argsort()]

            # 将种子本身也加入粘贴队列, 记录所有objs重塑后的尺寸
            all_wh = other_objs[:, -2:]
            scale = all_wh / self_wh
            scale = scale.max(1).reshape((-1, 1))
            all_wh = all_wh / scale
            all_wh = np.concatenate([all_wh, self_wh[None, :]]).astype(np.int64)
            num_all = num_other + 1

            all_whs0 = np.tile(all_wh[:, None, :], (1, num_try, 1)).astype(np.int64)
            dist_num = int(num_try * dist_num_rate)
            dist_matrix = np.random.uniform(dist_range[0], dist_range[-1], (dist_num, num_all, 2))
            dist_matrix = np.concatenate([dist_matrix, np.ones((num_try - dist_num, num_all, 2))], axis=0)
            np.random.shuffle(dist_matrix)
            dist_matrix = dist_matrix.transpose((1, 0, 2))
            all_whs1 = (all_whs0 * dist_matrix).astype(np.int64)
            # 第三次随机尺寸变换：随机挑选部分obj, 随机地进行整体放/缩
            scale_factor0 = np.ones((num_all, num_try // 2, 1))
            scale_factor1 = np.random.uniform(scale_range, 1., (num_all, num_try // 4, 1))
            scale_factor2 = np.random.uniform(1., 1. / scale_range, (num_all, num_try // 4, 1))
            scale_factor = np.concatenate([scale_factor0, scale_factor1, scale_factor2], axis=1)
            all_whs2 = (all_whs1 * scale_factor).astype(np.int64)

            # 随机挑选粘贴中心点
            all_xys = np.random.uniform(-paste_r, paste_r, (num_all, num_try, 2)) * all_whs2 + self_xywhs[i, :2].reshape((1, 1, 2))
            all_xywh = np.concatenate((all_xys, all_whs2), axis=-1).astype(np.int64)
            all_xyxy = xywh2xyxy_(all_xywh)
            clip_coords_(all_xyxy, (ih, iw))

            # 尺寸条件1：粘贴后的物体不能超出边界
            all_xywh_ = xyxy2xywh_(all_xyxy)
            all_whs_ = all_xywh_[..., 2:]
            size_mask0 = (all_whs_ / np.maximum(all_whs2, 1e-6)) == 1.
            # 尺寸条件2：裁剪物体时，绝对尺寸不得小于8×8
            size_mask1 = all_whs_ >= min_size
            size_mask = size_mask0 * size_mask1
            size_mask = size_mask[..., 0] * size_mask[..., 1]

            # 位置条件1：粘贴后的物体不能遮盖原来的目标
            overlap1, overlap2 = boxes_iou(all_xyxy.reshape((-1, 1, 4)), objs_xyxy[None, ...], split=True)[1:]
            overlap1, overlap2 = overlap1.reshape((num_all, num_try, -1)), overlap2.reshape((num_all, num_try, -1))
            overlap1_max, overlap2_max = overlap1.max(-1), overlap2.max(-1)
            overlap_mask = (overlap1_max <= max_overlap) * (overlap2_max <= max_overlap)
            mask = overlap_mask * size_mask
            num_proper = mask.astype(np.int64).sum(1)

            # 无符合条件的位置，结束copy-paste
            if num_proper.sum() == 0:
                continue

            # 随机挑选出符合尺寸条件与位置条件的box
            num_chosen = np.random.randint(1, max_chosen+1, num_all)
            num_chosen = np.minimum(num_proper, num_chosen)

            paste_objs = []
            indices = []
            for j in range(num_all):
                if num_chosen[j] > 0:
                    index = np.where(mask[j])
                    index = np.random.choice(index[0], num_chosen[j])
                    for k in index:
                        indices.append([j, k])
                    if len(paste_objs):
                        paste_objs = np.concatenate([paste_objs, all_xyxy[j][index]])
                    else:
                        paste_objs = all_xyxy[j][index]

            # 位置条件2：粘贴后的物体之间也不能互相遮盖
            overlap1, overlap2 = boxes_iou(paste_objs[:, None, :], paste_objs[None, ...], split=True)[1:]
            num_chosens = len(paste_objs)
            mask = np.eye(num_chosens, dtype=np.bool)
            overlap1[mask], overlap2[mask] = 0., 0.
            mask = (overlap1 > max_overlap) + (overlap2 > max_overlap)
            mask = mask.astype(np.int64)

            # 将不满足位置条件2的objs删除
            del_objs = []
            while np.any(mask):
                over_num = mask.sum(0)
                max_over = over_num.argmax()
                mask[max_over, :], mask[:, max_over] = 0, 0
                del_objs.append(max_over)
            # 无符合条件的位置，结束copy-paste
            if len(del_objs) == num_chosens:
                continue
            if len(del_objs):
                paste_objs = np.delete(paste_objs, del_objs, axis=0)
                indices = [item for i, item in enumerate(indices) if i not in del_objs]

            paste_objs_xywh = xyxy2xywh_(paste_objs)
            # 将objs粘贴, 过程包括: 选择并读取图像, 根据wh1进行crop, 根据wh3进行缩放，根据paste objs粘贴
            img_fore, img_id_fore = None, -2
            num_chosens = len(paste_objs)
            for j in range(num_chosens):
                # 选择并读取图像
                obj_id, obj_num = indices[j]
                # 在当前图像上选择obj
                if obj_id >= num_other:
                    ori_img = img.copy()
                    crop_xywh = self_xywhs[i]
                # 从其他图像上选择obj
                else:
                    img_id = int(other_objs[obj_id, 2])
                    crop_xywh = other_objs[obj_id, 3:]

                    if img_id == img_id_fore:
                        ori_img = img_fore.copy()
                    else:
                        ori_img = read_img(str(self.sub / self.obj_imgs[img_id]), way=READ_WAY, read_bgr=True)

                    img_fore = ori_img.copy()
                    img_id_fore = img_id

                scale_wh = all_whs2[obj_id, obj_num]
                paste_pos = paste_objs[j]

                # 建立crop box并在图像上进行crop
                crop_xywh = crop_xywh.reshape((-1, 4)).astype(np.int64)
                crop_xyxy = xywh2xyxy_(crop_xywh)
                clip_coords_(crop_xyxy, ori_img.shape)
                crop_xyxy = crop_xyxy[0]
                crop_obj = ori_img[crop_xyxy[1]: crop_xyxy[3], crop_xyxy[0]: crop_xyxy[2]]
                # 放缩
                scale_obj = cv2.resize(crop_obj, (scale_wh[0], scale_wh[1]))
                # 粘贴
                paste_img[paste_pos[1]: paste_pos[3], paste_pos[0]: paste_pos[2], :] = scale_obj

                objs_xyxy = np.concatenate([objs_xyxy, paste_objs], axis=0)
                # 将新的box添加到总boxes中，避免旧的粘贴obj被新的覆盖
                x, y, w, h = paste_objs_xywh[j]
                nx, ny, nw, nh = x / iw, y / ih, w / iw, h / ih
                paste_anno.append([obj_cls, nx, ny, nw, nh])
                paste_anno_abs.append([obj_cls, x, y, w, h])

        # img_ = paste_img.copy()
        # for j in range(num_objs):
        #     x, y, w, h = anno_abs[j][1:]
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # if paste_anno_abs:
        #     for j in range(len(paste_anno_abs)):
        #         x, y, w, h = paste_anno_abs[j][1:]
        #         x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #         box_ = [x1, y1, x2, y2]
        #         box_ = list(map(int, box_))
        #         cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 0, 255), 2)
        # cv2.imshow('2', img_)
        # cv2.waitKey()

        # 不要忘了将objs加入anno和anno abs中
        anno = anno + paste_anno
        anno_abs = anno_abs + paste_anno_abs
        return paste_img, anno, anno_abs

    # 将同类别物体组合成一个大型cluster然后整个粘贴到图像上
    def cluster_paste(self, img, anno, anno_abs):
        ih, iw = img.shape[:2]
        num_objs = len(anno_abs)
        objs = np.array(anno_abs).reshape((-1, 5))
        objs_xywh = objs[..., 1:]
        objs_xyxy = xywh2xyxy_(objs_xywh)

        # 剔除特殊类别
        if len(self.ban_cls):
            objs_cls = objs[:, 0].astype(np.int64)
            indices = []
            for i in range(num_objs):
                if objs_cls[i] not in self.ban_cls:
                    indices.append(i)
            num_objs = len(indices)
            if num_objs > 0:
                objs = objs[indices]
                objs_xywh = objs_xywh[indices]
            else:
                while num_objs == 0:
                    img, anno, anno_abs = self.easy_paste(img, anno, anno_abs, supplement=True)
                    num_objs = len(anno_abs)
                    objs = np.array(anno_abs).reshape((-1, 5))
                    objs_xywh = objs[..., 1:]
                    objs_xyxy = xywh2xyxy_(objs_xywh)
                    objs_cls = objs[:, 0].astype(np.int64)
                    indices = []
                    for i in range(num_objs):
                        if objs_cls[i] not in self.ban_cls:
                            indices.append(i)
                    num_objs = len(indices)
                objs = objs[indices]
                objs_xywh = objs_xywh[indices]

        crop_range = (0.9, 1.15)
        crop_num_rate = 0.5
        dist_range = (0.9, 1.1)
        dist_num_rate = 0.3  # 30%的obj进行长宽比例不一致的缩放，从而改变长宽比
        scale_range = 0.5  # 缩放倍数: 0.5——2.0
        min_size = 24  # 尺寸条件：粘贴的obj的边长不能小于该值
        max_overlap = 0.12  # obj与原目标以及obj之间的覆盖率阈值

        paste_r = 2.5  # 粘贴位置半径
        num_try = 200  # 每个obj随机生成200个粘贴位置
        max_try_num = 4  # 最大结晶次数
        num_sames = np.random.randint(5, 9, max_try_num)  # 每个粘贴种子附近的obj数(种子本身也在粘贴队列中)
        num_others = np.random.randint(1, 2, max_try_num)  # 每次结晶时添加的其他类别的obj数, 以增加真实性
        max_chosen = 2  # 每个obj最多粘贴几次
        max_all_num = 40  # 新增粘贴objs的上限
        min_all_num = 20  # 新增粘贴objs的下限, 低于此数认为没有结晶出有效的cluster

        """
        目标面积占图像总面积超过25%或目标总数超过150时
        每次结晶时再随机粘贴1-2个其他类别的物体

        1.先随机挑选几个目标作为种子/锚(粘贴围绕的中心)
        2.依次对每个种子随机选择同类别的粘贴物体，并且自身也加入粘贴队列中
        3.筛选得到每个粘贴中心周围的可粘贴位置(需要将粘贴物体的尺寸放缩到与原目标差不多)
        4.裁剪、缩放、粘贴，并加入标注
        """
        # 第一次从所有的目标中随机挑选一个种子
        self_index = random.randint(0, num_objs - 1)
        self_obj = objs[self_index]
        main_cls = self_obj[0]
        self_xywh = objs_xywh[self_index]
        self_wh = self_xywh[2:]
        cluster_center = self_xywh[:2].reshape((1, 1, 2))

        all_same_objs = self.objs[str(int(main_cls))]
        num_same_objs = self.obj_num[str(int(main_cls))]

        paste_anno, paste_anno_abs = [], []
        paste_img = img.copy()
        # 同时记录原位置尺寸(裁剪时需要坐标) 以及 重塑后的位置尺寸(粘贴时需要的尺寸)
        for i in range(max_try_num):
            num_same = num_sames[i]
            num_other = num_others[i]

            # 生成同类物体粘贴列表
            same_objs = []
            for j in range(num_same):
                same_obj = all_same_objs[random.randint(0, num_same_objs - 1)]
                same_objs.append([main_cls] + same_obj)
            same_objs.append([main_cls, -1] + self_xywh.tolist())
            num_same += 1
            same_objs = np.array(same_objs)
            same_objs = same_objs[same_objs[:, 2].argsort()]

            # 添加少量其他类别物体到粘贴列表中
            other_cls, other_objs = main_cls, []
            for j in range(num_other):
                while other_cls == main_cls:
                    other_cls = random.randint(0, self.num_cls - 1)
                other_obj = random.choice(self.objs[str(other_cls)])
                other_objs.append([other_cls] + other_obj)
            other_objs = np.array(other_objs)
            other_objs = other_objs[other_objs[:, 2].argsort()]

            all_objs = np.concatenate((same_objs, other_objs), axis=0)

            # 将种子本身也加入粘贴队列, 记录所有objs重塑后的尺寸
            all_wh = np.concatenate((same_objs[:, -2:], other_objs[:, -2:]), axis=0)
            scale = all_wh / (self_wh + 1e-6)
            scale = scale.max(1).reshape((-1, 1))
            all_wh = (all_wh / scale).astype(np.int64)
            num_all = num_same + num_other

            all_whs0 = np.tile(all_wh[:, None, :], (1, num_try, 1)).astype(np.int64)
            dist_num = int(num_try * dist_num_rate)
            dist_matrix = np.random.uniform(dist_range[0], dist_range[-1], (dist_num, num_all, 2))
            dist_matrix = np.concatenate([dist_matrix, np.ones((num_try - dist_num, num_all, 2))], axis=0)
            np.random.shuffle(dist_matrix)
            dist_matrix = dist_matrix.transpose((1, 0, 2))
            all_whs1 = (all_whs0 * dist_matrix).astype(np.int64)
            # 第三次随机尺寸变换：随机挑选部分obj, 随机地进行整体放/缩
            scale_factor0 = np.ones((num_all, num_try // 2, 1))
            scale_factor1 = np.random.uniform(scale_range, 1., (num_all, num_try // 4, 1))
            scale_factor2 = np.random.uniform(1., 1. / scale_range, (num_all, num_try // 4, 1))
            scale_factor = np.concatenate([scale_factor0, scale_factor1, scale_factor2], axis=1)
            all_whs2 = (all_whs1 * scale_factor).astype(np.int64)

            # 随机挑选粘贴中心点
            all_xys = np.random.uniform(-paste_r, paste_r, (num_all, num_try, 2)) * all_whs2 + cluster_center
            all_xywh = np.concatenate((all_xys, all_whs2), axis=-1).astype(np.int64)
            all_xyxy = xywh2xyxy_(all_xywh)
            clip_coords_(all_xyxy, (ih, iw))

            # 尺寸条件1：粘贴后的物体不能超出边界
            all_xywh_ = xyxy2xywh_(all_xyxy)
            all_whs_ = all_xywh_[..., 2:]
            size_mask0 = (all_whs_ / np.maximum(all_whs2, 1e-6)) == 1.
            # 尺寸条件2：裁剪物体时，绝对尺寸不得小于8×8
            size_mask1 = all_whs_ >= min_size
            size_mask = size_mask0 * size_mask1
            size_mask = size_mask[..., 0] * size_mask[..., 1]

            # 位置条件1：粘贴后的物体不能遮盖原来的目标
            overlap1, overlap2 = boxes_iou(all_xyxy.reshape((-1, 1, 4)), objs_xyxy[None, ...], split=True)[1:]
            overlap1, overlap2 = overlap1.reshape((num_all, num_try, -1)), overlap2.reshape((num_all, num_try, -1))
            overlap1_max, overlap2_max = overlap1.max(-1), overlap2.max(-1)
            overlap_mask = (overlap1_max <= max_overlap) * (overlap2_max <= max_overlap)
            mask = overlap_mask * size_mask
            num_proper = mask.astype(np.int64).sum(1)

            # 无符合条件的位置，结束copy-paste
            if num_proper.sum() == 0:
                continue

            # 随机挑选出符合尺寸条件与位置条件的box
            num_chosen = np.concatenate((np.random.randint(2, max_chosen + 1, num_other),
                                         np.random.randint(1, 3, num_same)))
            num_chosen = np.minimum(num_proper, num_chosen)

            paste_objs = []
            indices = []
            for j in range(num_all):
                if num_chosen[j] > 0:
                    index = np.where(mask[j])
                    index = np.random.choice(index[0], num_chosen[j])
                    for k in index:
                        indices.append([j, k])
                    if len(paste_objs):
                        paste_objs = np.concatenate([paste_objs, all_xyxy[j][index]])
                    else:
                        paste_objs = all_xyxy[j][index]

            # 位置条件2：粘贴后的物体之间也不能互相遮盖
            overlap1, overlap2 = boxes_iou(paste_objs[:, None, :], paste_objs[None, ...], split=True)[1:]
            num_chosens = len(paste_objs)
            mask = np.eye(num_chosens, dtype=np.bool)
            overlap1[mask], overlap2[mask] = 0., 0.
            mask = (overlap1 > max_overlap) + (overlap2 > max_overlap)
            mask = mask.astype(np.int64)

            # 将不满足位置条件2的objs删除
            del_objs = []
            while np.any(mask):
                over_num = mask.sum(0)
                max_over = over_num.argmax()
                mask[max_over, :], mask[:, max_over] = 0, 0
                del_objs.append(max_over)
            # 无符合条件的位置，结束copy-paste
            if len(del_objs) == num_chosens:
                continue
            if len(del_objs):
                paste_objs = np.delete(paste_objs, del_objs, axis=0)
                indices = [item for i, item in enumerate(indices) if i not in del_objs]

            paste_objs_xywh = xyxy2xywh_(paste_objs)
            # 将objs粘贴, 过程包括: 选择并读取图像, 根据wh1进行crop, 根据wh3进行缩放，根据paste objs粘贴
            img_fore, img_id_fore = None, -2
            num_chosens = len(paste_objs)
            for j in range(num_chosens):
                # 选择并读取图像
                obj_id, obj_num = indices[j]
                cls = all_objs[obj_id][0]
                crop_xywh = all_objs[obj_id, 2:]
                img_id = int(all_objs[obj_id, 1])
                if img_id == img_id_fore:
                    ori_img = img_fore.copy()
                else:
                    if img_id == -1:
                        ori_img = img.copy()
                    else:
                        ori_img = read_img(str(self.sub / self.obj_imgs[img_id]), way=READ_WAY, read_bgr=True)
                img_fore = ori_img.copy()
                img_id_fore = img_id

                scale_wh = all_whs2[obj_id, obj_num]
                paste_pos = paste_objs[j]

                # 建立crop box并在图像上进行crop
                crop_xywh = crop_xywh.reshape((-1, 4)).astype(np.int64)
                crop_xyxy = xywh2xyxy_(crop_xywh)
                clip_coords_(crop_xyxy, ori_img.shape)
                crop_xyxy = crop_xyxy[0]
                crop_obj = ori_img[crop_xyxy[1]: crop_xyxy[3], crop_xyxy[0]: crop_xyxy[2]]
                # 放缩
                scale_obj = cv2.resize(crop_obj, (scale_wh[0], scale_wh[1]))
                # 粘贴
                paste_img[paste_pos[1]: paste_pos[3], paste_pos[0]: paste_pos[2], :] = scale_obj

                # 将新的box添加到总boxes中，避免旧的粘贴obj被新的覆盖
                objs_xyxy = np.concatenate([objs_xyxy, paste_objs], axis=0)

                x, y, w, h = paste_objs_xywh[j]
                nx, ny, nw, nh = x / iw, y / ih, w / iw, h / ih
                paste_anno.append([cls, nx, ny, nw, nh])
                paste_anno_abs.append([cls, x, y, w, h])

            if len(paste_anno) > max_all_num:
                break

            # 从结晶中再随机选择一个新的结晶中心
            cluster_center = paste_objs_xywh[random.randint(0, num_chosens-1), :2].reshape((1, 1, 2))

        # img_ = paste_img.copy()
        # for j in range(num_objs):
        #     x, y, w, h = anno_abs[j][1:]
        #     x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #     box_ = [x1, y1, x2, y2]
        #     box_ = list(map(int, box_))
        #     cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 255, 0), 2)
        # if paste_anno_abs:
        #     for j in range(len(paste_anno_abs)):
        #         x, y, w, h = paste_anno_abs[j][1:]
        #         x1, y1, x2, y2 = x - w / 2., y - h / 2., x + w / 2., y + h / 2.
        #         box_ = [x1, y1, x2, y2]
        #         box_ = list(map(int, box_))
        #         cv2.rectangle(img_, (box_[0], box_[1]), (box_[2], box_[3]), (0, 0, 255), 2)
        # cv2.imshow('2', img_)
        # cv2.waitKey()

        # 不要忘了将objs加入anno和anno abs中
        anno = anno + paste_anno
        anno_abs = anno_abs + paste_anno_abs
        return paste_img, anno, anno_abs


def split(img, patch_w, patch_h, num_patches_w, num_patches_h):
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
    bias_x, bias_y = bias_x.flatten(), bias_y.flatten()
    bias = np.stack([bias_x, bias_y, bias_x, bias_y], axis=-1)
    bias = bias[:, None, :]
    bias = torch.from_numpy(bias).cuda().type(torch.float32)

    sub_imgs = []
    for j in range(num_patches_h):
        for i in range(num_patches_w):
            sub_img = img[patches_y1[j]: patches_y2[j], patches_x1[i]: patches_x2[i]]
            sub_imgs.append(sub_img)

    return sub_imgs, bias


def split_with_anno(img, anno, patch_w, patch_h, num_patches_w, num_patches_h, with_ori=False):
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
    bias_x, bias_y = bias_x.flatten(), bias_y.flatten()
    bias = np.stack([bias_x, bias_y, bias_x, bias_y], axis=-1)
    bias = bias[:, None, :]
    bias = torch.from_numpy(bias).cuda().type(torch.float32)

    sub_imgs = []
    for j in range(num_patches_h):
        for i in range(num_patches_w):
            sub_img = img[patches_y1[j]: patches_y2[j], patches_x1[i]: patches_x2[i]]
            sub_imgs.append(sub_img)

    sub_annos = []
    if anno is None:
        return sub_imgs, bias, None

    else:
        num_objs = len(anno)
        for i in range(num_objs):
            cls = anno[i, 0]
            x, y, w, h = anno[i, 1:]
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
                            if with_ori and ((nw > MAX_W) or (nh > MAX_H)):
                                continue
                            else:
                                a = 0
                            sub_annos.append([patch_y * num_patches_h + patch_x, cls, nx, ny, nw, nh])
                        else:
                            a = 1
        if len(sub_annos) > 0:
            sub_annos = np.array(sub_annos)
            sub_annos = torch.from_numpy(sub_annos).cuda().type(torch.float32)
            sort = sub_annos[:, 0].argsort()
            sub_annos = sub_annos[sort]
    return sub_imgs, bias, sub_annos


class LoadVal:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, root, sub, img_size=640, stride=32, auto=False, batch_sz=1, with_ori=False, return_targets=False,
                 patch_w=None, patch_h=None, num_patches_w=None, num_patches_h=None):
        with open(str(root / (sub + '.json')), 'r') as f:
            data = json.load(f)
            f.close()
        if '-sampled' in sub:
            sub = sub[:sub.index('-sampled')]
        if os.path.isdir(root / sub / 'images'):
            self.sub = root / sub / 'images'
        elif os.path.isdir(root / 'images' / sub):
            self.sub = root / 'images' / sub

        self.with_ori = with_ori
        self.return_targets = return_targets

        if patch_w is None:
            patch_w = PATCH_W
        if patch_h is None:
            patch_h = PATCH_H
        if num_patches_w is None:
            num_patches_w = NUM_PATCHES_W
        if num_patches_h is None:
            num_patches_h = NUM_PATCHES_H
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.num_patches_w = num_patches_w
        self.num_patches_h = num_patches_h
        self.num_patches = self.num_patches_w * self.num_patches_h

        self.imgs = data['imgs']
        self.num_imgs = len(self.imgs)
        self.annos = data['labels']
        labels = data['labels']
        for i in range(self.num_imgs):
            labels[i] = np.array(labels[i])
        self.labels = labels
        shapes = data['shapes']
        shapes = np.array(shapes, dtype=np.float64)
        self.shapes = shapes

        self.img_size = img_size
        self.stride = stride
        self.auto = auto

    def __len__(self):
        return self.num_imgs  # number of files

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.num_imgs:
            raise StopIteration

        img_name = self.imgs[self.count]
        labels = self.labels[self.count].copy()

        self.count += 1
        img0 = read_img(str(self.sub / img_name), way=READ_WAY, read_bgr=True)

        # sub_imgs, bias = split(img0, self.patch_w, self.patch_h, self.num_patches_w, self.num_patches_h)
        targets = None if not self.return_targets else labels
        sub_imgs, bias, targets = split_with_anno(img0, targets, self.patch_w, self.patch_h,
                                                  self.num_patches_w, self.num_patches_h, with_ori=self.with_ori)
        shape0 = sub_imgs[0].shape
        sub_imgs = letterbox_batch(sub_imgs, self.num_patches, self.img_size, stride=self.stride, auto=self.auto)[0]

        ori_shapes = [img0.shape[:2]]
        if self.with_ori:
            ori_img = img0.copy()
            ori_img, ratio, pad = letterbox(ori_img, self.img_size, stride=self.stride, auto=self.auto)
            sub_imgs.append(ori_img)
            ori_shapes.append(ratio)
            ori_shapes.append(pad)

            if self.return_targets:
                ori_targets = []
                for i in range(labels.shape[0]):
                    cls, x, y, w, h = labels[i]
                    if (MIN_W <= w < 1.) and (MIN_H <= h < 1.):
                        ori_targets.append([self.num_patches, cls, x, y, w, h])
                if len(ori_targets) > 0:
                    ori_targets = np.array(ori_targets)
                    ori_targets = torch.from_numpy(ori_targets).cuda().type(torch.float32)
                    if targets != []:
                        targets = torch.cat([targets, ori_targets], dim=0)
                    else:
                        targets = ori_targets

        img = np.array(sub_imgs)

        # Convert
        # HWC to CHW, BGR to RGB
        img = np.stack([img[..., 2], img[..., 1], img[..., 0]], axis=-1)
        img = img.transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)

        return img, ori_shapes, labels, bias, shape0, str(self.sub / img_name), img0, targets


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True, batch_sz=1, with_ori=False,
                 patch_w=None, patch_h=None, num_patches_w=None, num_patches_h=None):
        if patch_w is None:
            patch_w = PATCH_W
        if patch_h is None:
            patch_h = PATCH_H
        if num_patches_w is None:
            num_patches_w = NUM_PATCHES_W
        if num_patches_h is None:
            num_patches_h = NUM_PATCHES_H
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.num_patches_w = num_patches_w
        self.num_patches_h = num_patches_h
        self.num_patches = self.num_patches_w * self.num_patches_h

        self.with_ori = with_ori

        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __len__(self):
        return self.nf  # number of files

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = read_img(path, way=READ_WAY, read_bgr=True)
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        sub_imgs, bias = split_with_anno(img0, None, self.patch_w, self.patch_h,
                                         self.num_patches_w, self.num_patches_h, with_ori=self.with_ori)[:2]
        shape0 = sub_imgs[0].shape
        sub_imgs = letterbox_batch(sub_imgs, self.num_patches, self.img_size, stride=self.stride, auto=self.auto)[0]

        if self.with_ori:
            ori_img = img0.copy()
            ori_img = letterbox(ori_img, self.img_size, stride=self.stride, auto=self.auto)[0]
            sub_imgs.append(ori_img)

        img = np.array(sub_imgs)

        # Convert
        # HWC to CHW, BGR to RGB
        img = np.stack([img[..., 2], img[..., 1], img[..., 0]], axis=-1)
        img = img.transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, s, bias, shape0

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


def letterbox_batch(imgs, num_img, new_shape=(640, 640), color=(114, 114, 114),
                    auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = imgs[0].shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if shape[::-1] != new_unpad:  # resize
        for i in range(num_img):
            img = cv2.resize(imgs[i], new_unpad, interpolation=cv2.INTER_LINEAR)
            imgs[i] = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return imgs, ratio, (dw, dh)


def xyxy2xywh_(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy_(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def scale_coords_(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[..., [0, 2]] -= pad[0]  # x padding
    coords[..., [1, 3]] -= pad[1]  # y padding
    coords[..., :4] /= gain
    clip_coords_(coords, img0_shape)
    return coords


def clip_coords_(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def boxes_iou(box1, box2, split=False):
    box1_area = np.maximum(np.prod((box1[..., 2:] - box1[..., :2]), axis=-1), 1e-6)
    box2_area = np.maximum(np.prod((box2[..., 2:] - box2[..., :2]), axis=-1), 1e-6)

    left_top = np.maximum(box1[..., :2], box2[..., :2])
    right_bottom = np.minimum(box1[..., 2:], box2[..., 2:])
    inter = np.maximum(right_bottom - left_top, 0.)
    inter_are = np.prod(inter, axis=-1)

    iou = inter_are / (box1_area + box2_area - inter_are)
    if split:
        iou1 = inter_are / box1_area
        iou2 = inter_are / box2_area
        return iou, iou1, iou2
    else:
        return iou


def create_train_loader(root, sub, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False,
                        pad=0.0, rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='',
                        shuffle=False):
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = Generator(root, sub, imgsz, batch_size,
                            augment=augment,  # augmentation
                            hyp=hyp,  # hyperparameters
                            rect=rect,  # rectangular batches
                            cache_images=cache,
                            single_cls=single_cls,
                            stride=int(stride),
                            pad=pad,
                            image_weights=image_weights,
                            prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // WORLD_SIZE, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle and sampler is None,
                      num_workers=nw,
                      sampler=sampler,
                      pin_memory=True,
                      collate_fn=collate_fn4 if quad else collate_fn), dataset


def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).type(torch.uint8)
    labels = torch.cat((labels, labels2), 0)
    return im, labels


def box_iou(box1, box2):
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    left_up = np.maximum(box1[..., :2], box2[..., :2])
    right_down = np.minimum(box1[..., 2:], box2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou


def collate_fn(batch):
    img, label, path, shapes = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def collate_fn4(batch):
    img, label, path, shapes = zip(*batch)  # transposed
    n = len(shapes) // 4
    img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

    ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
    wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
    s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
    for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
        i *= 4
        if random.random() < 0.5:
            im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[
                0].type(img[i].type())
            l = label[i]
        else:
            im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
            l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
        img4.append(im)
        label4.append(l)

    for i, l in enumerate(label4):
        l[:, 0] = i  # add target image index for build_targets()

    return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# if __name__ == "__main__":
#     root = Path('F://DataBase/CarData/')
#     for sub in 'train', 'test', 'val':
#         # a = Generator(root, sub)
#         a = Generator(root, sub)
#         for k in range(1000):
#             # a.mosaic2_vertical(k)
#             # a.mosaic2_horizontal(k)
#             # a.mosaic4(k)
#             b = a.random_crop(k, 0.8, 0.8)
#             if b:
#                 c = a.random_paste(b[0], b[1], b[2])
#     pass
