import os

import numpy as np
import cv2
import torch


from utils.plots import Annotator, colors


def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False


class Visualizer(object):
    def __init__(self, class_names):
        self.class_names = class_names

    def pred_vis(self, img, pred_, img_name, waitkey=None, vis=False):
        annotator = Annotator(img, line_width=2, example=str(self.class_names))

        pred = pred_.detach().cpu().numpy()
        num_objs = pred.shape[0]
        boxes, conf_out, cls_out = pred[:, :4], pred[:, 4], pred[:, -1]

        for i in range(num_objs):
            c = int(cls_out[i])
            label = f'{self.class_names[c]} {conf_out[i]:.2f}'
            annotator.box_label(boxes[i], label, color=colors(c, True))
        result_img = annotator.result()
        result_img = np.ascontiguousarray(result_img).astype(np.uint8)

        if vis:
            cv2.imshow(img_name, result_img)
            if waitkey is None:
                cv2.waitKey()
            else:
                cv2.waitKey(waitkey)
        return result_img


def make_name(opt):
    w_name = opt.name + '--weight'

    if isinstance(opt.weights, list):
        for w in opt.weights:
            name = w.split('/')[-1].split('.')[0]
            w_name += '-[{}]'.format(name)
    else:
        name = opt.weights.split('/')[-1].split('.')[0]
        w_name += '-[{}]'.format(name)

    if hasattr(opt, 'val_sub'):
        data_name = opt.val_sub
    elif hasattr(opt, 'source'):
        data_name = opt.source
        data_name = data_name.split('/')[-3]

    if isinstance(opt.imgsz, int):
        imgsz_name = opt.imgsz
    else:
        imgsz_name = opt.imgsz[0]

    ori_name = 'T' if opt.with_ori else 'F'
    aug_name = 'T' if opt.augment else 'F'
    half_name = 'T' if opt.half else 'F'

    output_name = w_name + '--data-[{}]--imgsz-{}--ori-{}--aug-{}--half-{}'. \
        format(data_name, imgsz_name, ori_name, aug_name, half_name)

    test_name = output_name + '--nms-{}'.format(opt.nms_type)

    vis_name = test_name + '--conf-{:.7f}--iou-{:.7f}'.format(opt.conf_thres, opt.iou_thres)

    return output_name, test_name, vis_name
