

import json
from PIL import Image

from utils.general import os, Path


def build_json(root, sub):
    imgs_path = root / 'images' / sub
    annos_path = root / 'labels' / sub
    imgs = os.listdir(imgs_path)
    num_imgs = len(imgs)

    json_file = {}
    data = {}
    shape = {}
    imgs_ = []
    for i in range(num_imgs):
        img_name = imgs[i]
        anno_name = img_name.split('.')[0] + '.txt'
        img = Image.open(str(imgs_path / img_name))
        # img = cv2.imread(str(imgs_path / img_name))
        with open(str(annos_path / anno_name), 'r') as f:
            anno = f.readlines()
            f.close()
        num_objs = len(anno)
        if num_objs:
            for j in range(num_objs):
                anno[j] = list(map(float, anno[j].split()))
            data[img_name] = anno
            shape[img_name] = img.size
            imgs_.append(img_name)
    json_file['imgs'] = imgs_
    json_file['labels'] = data
    json_file['shapes'] = shape
    json_file['dir'] = sub + '/images'

    file = json.dumps(json_file, indent=4)
    fileObject = open(str(root / sub) + '.json', 'w')
    fileObject.write(file)
    fileObject.close()
    pass


def build_json_visdrone(root, sub):
    imgs_path = root / sub / 'images'
    annos_path = root / sub / 'labels'
    annos = os.listdir(annos_path)
    num_imgs = len(annos)

    json_file = {}
    data = []
    shape = []
    imgs_ = []
    data_abs = []
    nan = 0
    for i in range(num_imgs):
        anno_name = annos[i]
        img_name = anno_name.split('.')[0] + '.jpg'

        img = Image.open(str(imgs_path / img_name))
        with open(str(annos_path / anno_name), 'r') as f:
            anno = f.readlines()
            f.close()
        num_objs = len(anno)
        if num_objs:
            anno_abs = []
            iw, ih = img.size
            for j in range(num_objs):
                anno[j] = list(map(float, anno[j].split()))
                anno_abs.append([anno[j][0],
                                 int(anno[j][1] * iw), int(anno[j][2] * ih),
                                 int(anno[j][3] * iw), int(anno[j][4] * ih)])
            data.append(anno)
            shape.append(img.size)
            data_abs.append(anno_abs)
            imgs_.append(img_name)
        else:
            nan += 1

    json_file['imgs'] = imgs_
    json_file['labels'] = data
    json_file['labels_abs'] = data_abs
    json_file['shapes'] = shape
    json_file['dir'] = '/images/' + sub

    file = json.dumps(json_file, indent=4)
    fileObject = open(str(root / sub) + '.json', 'w')
    fileObject.write(file)
    fileObject.close()


def build_json_visdrone_cls(root, sub):
    cls_dict = ['pedestrian', 'people', 'bicycle', 'car', 'van',
                'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    cls_num, cls_data = {}, {}
    for i in range(10):
        cls_num[i] = 0
        cls_data[i] = []

    imgs_path = root / sub / 'images'
    annos_path = root / sub / 'annotations'
    annos = os.listdir(annos_path)
    num_imgs = len(annos)

    json_file = {}
    data = {}
    shape = {}
    imgs_ = []

    nan = 0
    for i in range(num_imgs):
        anno_name = annos[i]
        img_name = anno_name.split('.')[0] + '.jpg'

        img = Image.open(str(imgs_path / img_name))
        # img = cv2.imread(str(imgs_path / img_name))
        with open(str(annos_path / anno_name), 'r') as f:
            anno = f.readlines()
            f.close()
        num_objs = len(anno)

        new_anno = []
        if num_objs:
            im_sz = img.size
            for j in range(num_objs):
                anno_j = list(map(int, anno[j].split(',')[:8]))
                if anno_j[4] == 0:
                    continue

                # if anno_j[6] > 1 or anno_j[7] > 1:
                if anno_j[6] > 0 or anno_j[7] > 0:
                    continue

                if anno_j[0] < 0 or anno_j[1] < 0 or anno_j[2] <= 0 or anno_j[3] <= 0:
                    continue

                if anno_j[2] <= 12 or anno_j[3] <= 12:
                    continue

                if anno_j[0] + anno_j[2] > im_sz[0]:
                    anno_j[2] = im_sz[0] - anno_j[0] - 1

                if anno_j[1] + anno_j[3] > im_sz[1]:
                    anno_j[3] = im_sz[1] - anno_j[1] - 1

                x0, y0, w, h = anno_j[:4]
                x, y = x0 + w // 2, y0 + h // 2
                cls = anno_j[5] - 1
                cls_num[cls] += 1
                cls_data[cls].append([i, x, y, w, h])

                new_anno.append(anno_j)

            data[i] = new_anno
            shape[i] = im_sz
            imgs_.append(img_name)
        else:
            nan += 1

    json_file['imgs'] = imgs_
    # json_file['labels'] = data
    # json_file['shapes'] = shape
    json_file['dir'] = '/images/' + sub
    json_file['objs'] = cls_data
    json_file['cls_num'] = cls_num
    json_file['cls_dict'] = cls_dict

    file = json.dumps(json_file, indent=4)
    fileObject = open(str(root / sub) + '-cls.json', 'w')
    fileObject.write(file)
    fileObject.close()


def build_json_uavdt_cls(root, sub):
    cls_dict = ['car', 'truck', 'bus']
    cls_num, cls_data = {}, {}
    for i in range(3):
        cls_num[i] = 0
        cls_data[i] = []

    imgs_path = root / sub / 'images'
    annos_path = root / sub / 'annotations'
    annos = os.listdir(annos_path)
    num_imgs = len(annos)

    json_file = {}
    data = {}
    shape = {}
    imgs_ = []

    nan = 0
    for i in range(num_imgs):
        anno_name = annos[i]
        img_name = anno_name.split('.')[0] + '.jpg'

        img = Image.open(str(imgs_path / img_name))
        # img = cv2.imread(str(imgs_path / img_name))
        with open(str(annos_path / anno_name), 'r') as f:
            anno = f.readlines()
            f.close()
        num_objs = len(anno)

        new_anno = []
        if num_objs:
            im_sz = img.size
            for j in range(num_objs):
                anno_j = list(map(int, anno[j].split(',')[:8]))
                if anno_j[4] == 0:
                    continue

                if anno_j[6] > 1 or anno_j[7] > 1:
                    continue

                if anno_j[0] < 0 or anno_j[1] < 0 or anno_j[2] <= 0 or anno_j[3] <= 0:
                    continue

                if anno_j[2] <= 12 or anno_j[3] <= 12:
                    continue

                if anno_j[0] + anno_j[2] > im_sz[0]:
                    anno_j[2] = im_sz[0] - anno_j[0] - 1

                if anno_j[1] + anno_j[3] > im_sz[1]:
                    anno_j[3] = im_sz[1] - anno_j[1] - 1

                x0, y0, w, h = anno_j[:4]
                x, y = x0 + w // 2, y0 + h // 2
                cls = anno_j[5]
                cls_num[cls] += 1
                cls_data[cls].append([i, x, y, w, h])

                new_anno.append(anno_j)

            data[i] = new_anno
            shape[i] = im_sz
            imgs_.append(img_name)
        else:
            nan += 1

    json_file['imgs'] = imgs_
    # json_file['labels'] = data
    # json_file['shapes'] = shape
    json_file['dir'] = '/images/' + sub
    json_file['objs'] = cls_data
    json_file['cls_num'] = cls_num
    json_file['cls_dict'] = cls_dict

    file = json.dumps(json_file, indent=4)
    fileObject = open(str(root / sub) + '-cls.json', 'w')
    fileObject.write(file)
    fileObject.close()


if __name__ == "__main__":
    root = Path('/home/Data/Visdrone2019-DET/')
    for sub in ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']:
        build_json_visdrone(root, sub)

    root = Path('/home/Data/Visdrone2019-DET/')
    for sub in ['VisDrone2019-DET-train']:
        build_json_visdrone_cls(root, sub)

    # root = Path('/home/Data/UAVDT/')
    # for sub in ['UAVDT-train', 'UAVDT-test']:
    #     build_json_visdrone(root, sub)

    # root = Path('/home/Data/UAVDT/')
    # for sub in ['UAVDT-train']:
    #     build_json_uavdt_cls(root, sub)
