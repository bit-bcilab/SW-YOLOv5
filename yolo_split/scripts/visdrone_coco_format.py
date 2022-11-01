

import json
import os
import cv2
import numpy as np
from PIL import Image
import shutil


class Vis2COCO:
    def __init__(self, save_path, train_ratio, category_list, is_mode="train"):
        self.category_list = category_list
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.is_mode = is_mode
        self.train_ratio = train_ratio
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def to_coco(self, anno_dir, img_dir):
        self._init_categories()
        img_list = os.listdir(img_dir)
        for img_name in img_list:
            anno_path = os.path.join(anno_dir, img_name.replace(os.path.splitext(img_name)[-1], '.txt'))
            if not os.path.isfile(anno_path):
                print('File is not exist!', anno_path)
                continue

            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            h, w, c = img.shape
            self.images.append(self._image(img_path, h, w))
            if self.img_id % 500 == 0:
                print("处理到第{}张图片".format(self.img_id))

            with open(anno_path, 'r') as f:
                for lineStr in f.readlines():
                    try:
                        if ',' in lineStr:
                            xmin, ymin, w, h, score, category, trunc, occlusion = lineStr.split(',')
                        else:
                            xmin, ymin, w, h, score, category, trunc, occlusion = lineStr.split()
                    except:
                        # print('error: ', anno_path, 'line: ', lineStr)
                        continue
                    # if int(category) in [0, 11] or int(w) < 4 or int(h) < 4:
                    #     continue
                    if int(category) in [0, 11] or int(trunc) > 1 or int(occlusion) > 1:
                        continue
                    if int(w) < 4 or int(h) < 4:
                        a = 1
                    label, bbox = int(category), [int(xmin), int(ymin), int(w), int(h)]
                    annotation = self._annotation(label, bbox, img_path)
                    self.annotations.append(annotation)
                    self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'VisDrone'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        cls_num = len(self.category_list)
        for v in range(1, cls_num + 1):
            # print(v)
            category = {}
            category['id'] = v
            category['name'] = self.category_list[v - 1]
            category['supercategory'] = self.category_list[v - 1]
            self.categories.append(category)

    def _image(self, path, h, w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = path.split(os.sep)[-1].split('.')[0]
        # image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    def _annotation(self, label, bbox, path=None):
        area = bbox[2] * bbox[3]
        annotation = {}
        annotation['id'] = self.ann_id
        if path is None:
            annotation['image_id'] = self.img_id
        else:
            annotation['image_id'] = path.split(os.sep)[-1].split('.')[0]
        annotation['category_id'] = label
        annotation['segmentation'] = []
        annotation['bbox'] = bbox
        annotation['iscrowd'] = 0
        annotation["ignore"] = 0
        annotation['area'] = area
        return annotation

    def save_coco_json(self, instance, save_path):
        import json
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=4, separators=(',', ': '))


def cvt_vis2coco(img_path, anno_path, save_path, train_ratio=0.9, category_list=[], mode='train'):  # mode: train or val
    vis2coco = Vis2COCO(save_path, train_ratio, category_list, is_mode=mode)
    train_instance = vis2coco.to_coco(anno_path, img_path)
    if not os.path.exists(os.path.join(save_path, "Anno")):
        os.makedirs(os.path.join(save_path, "Anno"))
    vis2coco.save_coco_json(train_instance,
                            os.path.join(save_path, 'Anno', 'VisDrone2019-DET_{}_coco.json'.format(mode)))
    print('Process {} Done'.format(mode))


class UAVDT2COCO:
    def __init__(self, save_path, train_ratio, category_list, is_mode="train"):
        self.category_list = category_list
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.is_mode = is_mode
        self.train_ratio = train_ratio
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def to_coco(self, anno_dir, img_dir):
        self._init_categories()
        img_list = os.listdir(img_dir)
        for img_name in img_list:
            anno_path = os.path.join(anno_dir, img_name.replace(os.path.splitext(img_name)[-1], '.txt'))
            if not os.path.isfile(anno_path):
                print('File is not exist!', anno_path)
                continue

            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            h, w, c = img.shape
            self.images.append(self._image(img_path, h, w))
            if self.img_id % 500 == 0:
                print("处理到第{}张图片".format(self.img_id))

            with open(anno_path, 'r') as f:
                for lineStr in f.readlines():
                    try:
                        if ',' in lineStr:
                            xmin, ymin, w, h, score, category, trunc, occlusion = lineStr.split(',')
                        else:
                            xmin, ymin, w, h, score, category, trunc, occlusion = lineStr.split()
                    except:
                        # print('error: ', anno_path, 'line: ', lineStr)
                        continue
                    # if int(category) in [0, 11] or int(w) < 4 or int(h) < 4:
                    #     continue
                    if int(score) == 0 or int(trunc) > 1 or int(occlusion) > 1:
                        continue
                    if int(w) < 4 or int(h) < 4:
                        a = 1
                    label, bbox = int(category) + 1, [int(xmin), int(ymin), int(w), int(h)]
                    annotation = self._annotation(label, bbox, img_path)
                    self.annotations.append(annotation)
                    self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'UAVDT'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        cls_num = len(self.category_list)
        for v in range(1, cls_num + 1):
            # print(v)
            category = {}
            category['id'] = v
            category['name'] = self.category_list[v - 1]
            category['supercategory'] = self.category_list[v - 1]
            self.categories.append(category)

    def _image(self, path, h, w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = path.split(os.sep)[-1].split('.')[0]
        # image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    def _annotation(self, label, bbox, path=None):
        area = bbox[2] * bbox[3]
        annotation = {}
        annotation['id'] = self.ann_id
        if path is None:
            annotation['image_id'] = self.img_id
        else:
            annotation['image_id'] = path.split(os.sep)[-1].split('.')[0]
        annotation['category_id'] = label
        annotation['segmentation'] = []
        annotation['bbox'] = bbox
        annotation['iscrowd'] = 0
        annotation["ignore"] = 0
        annotation['area'] = area
        return annotation

    def save_coco_json(self, instance, save_path):
        import json
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=4, separators=(',', ': '))


def cvt_uavdt2coco(img_path, anno_path, save_path, train_ratio=0.9, category_list=[], mode='train'):  # mode: train or val
    uavdt2coco = UAVDT2COCO(save_path, train_ratio, category_list, is_mode=mode)
    train_instance = uavdt2coco.to_coco(anno_path, img_path)
    if not os.path.exists(os.path.join(save_path, "Anno")):
        os.makedirs(os.path.join(save_path, "Anno"))
    uavdt2coco.save_coco_json(train_instance,
                              os.path.join(save_path, 'Anno', 'UAVDT_{}_coco.json'.format(mode)))
    print('Process {} Done'.format(mode))


if __name__ == "__main__":
    # examples_write_json()
    # root_path = 'D://DataBase/VisDrone2019-DET/'
    # category_list = ['pedestrian', 'people', 'bicycle', 'car', 'van',
    #                  'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    # for mode in ['val', 'test-dev']:  # 'train',
    #     cvt_vis2coco(os.path.join(root_path, 'VisDrone2019-DET-{}/images'.format(mode)),
    #                  os.path.join(root_path, 'VisDrone2019-DET-{}/annotations'.format(mode)),
    #                  root_path, category_list=category_list, mode=mode)

    root_path = '/home/Data2/UAVDT/'
    category_list = ['car', 'truck', 'bus']
    for mode in ['test']:
        cvt_uavdt2coco(os.path.join(root_path, 'UAVDT-{}/images'.format(mode)),
                       os.path.join(root_path, 'UAVDT-{}/annotations'.format(mode)),
                       root_path, category_list=category_list, mode=mode)
