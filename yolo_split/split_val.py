# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread
import cv2
import numpy as np
from tqdm import tqdm
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh, xywhn2xyxy, xyxy2xywhn)
from utils.metrics import ConfusionMatrix, ap_per_class, ap_per_class_modified
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from utils.loss import ComputeLoss

from yolo_split.split_dataset import LoadVal, scale_coords_, xyxy2xywh_, xywh2xyxy_
from yolo_split.config import PATCH_SETTINGS, SCALE_SETTINGS, SCALE_SETTINGS_VAL

from yolo_split.split_utils.tools import str2bool, Visualizer, make_name

import logging


logger = logging.getLogger('global')

MODE = 'Visdrone'
# MODE = 'UAVDT'
PATCH_W, PATCH_H, NUM_PATCHES_W, NUM_PATCHES_H = PATCH_SETTINGS[MODE]
overlap_w = (NUM_PATCHES_W * PATCH_W - 1.) / (NUM_PATCHES_W - 1.)
overlap_h = (NUM_PATCHES_H * PATCH_H - 1.) / (NUM_PATCHES_H - 1.)
MIN_W, MIN_H, MAX_W, MAX_H = SCALE_SETTINGS_VAL[MODE]
MIN_W = overlap_w if MIN_W > overlap_w else MIN_W
MIN_H = overlap_h if MIN_H > overlap_h else MIN_H


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5]) + 1],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataset=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        val_sub='val',
        with_ori=False,
        nms_type='default',
        post_cls=False,
        vis=False
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by split_train_tph.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()

        if compute_loss is None:
            compute_loss = ComputeLoss(model)
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()
        elif engine:
            batch_size = model.batch_size
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device('cpu')
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    lp = 4 + 1 + nc
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    if dataset is None:
        dataset = LoadVal(Path(data['path']), val_sub, imgsz, model.stride, auto=False,
                          batch_sz=batch_size, with_ori=with_ori, return_targets=True)  #(compute_loss is not None)

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))  # list(range(1, 1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    jdict, stats, ap, ap_class = [], [], [], []
    loss = torch.zeros(3, device=device)

    viser = Visualizer(class_names=names)

    plot_num = 4  # 9/16
    plot_batch = 3
    plot_label = plots
    ims = []
    targets_ = []
    out = []
    paths_ = []

    num_loss = 0
    pbar = tqdm(dataset, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for batch, (im, ori_shapes, labels, bias, shape0, paths, ori_img, targets) in enumerate(pbar):
        valid_target = False

        ih, iw = ori_shapes[0]
        input_size = im.shape[2:]
        if targets is not None and targets != []:
            targets = targets.to(device)
            valid_target = True
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # pred = model(im, augment=augment, visualize=False)
        pred, train_out = model(im) if training else model(im, augment=augment, val=True)
        t3 = time_sync()
        dt[1] += t3 - t2

        if valid_target and compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls
            num_loss += 1

        if not with_ori:
            pred[..., :4] = xywh2xyxy_(pred[..., :4])
            pred[..., :4] = scale_coords_(input_size, pred[..., :4], shape0).round()
            pred[..., :4] = pred[..., :4] + bias
        # 分开处理原始图像与切块图像的推理结果
        else:
            # 原始图像上的推理结果的筛选：只保存相对尺寸大于设定阈值的结果
            ori_pred = pred[-1, :, :5]
            sw_pred = pred[:-1, :, :5]
            tiny_pred_index = (ori_pred[..., 2] < (input_size[1] * MIN_W)) * (ori_pred[..., 3] < (input_size[0] * MIN_H))
            ori_pred[..., -1][tiny_pred_index] = 0.
            big_pred_index = (sw_pred[..., 2] > (input_size[1] * MAX_W * PATCH_W)) + (sw_pred[..., 3] > (input_size[0] * MAX_H * PATCH_H))
            sw_pred[..., -1][big_pred_index] = 0.
            pred[-1, :, :5] = ori_pred
            pred[:-1, :, :5] = sw_pred

            pred[..., :4] = xywh2xyxy_(pred[..., :4])
            pred[-1, :, :4] = scale_coords_(input_size, pred[-1, :, :4], (ih, iw)).round()
            pred[:-1, :, :4] = scale_coords_(input_size, pred[:-1, :, :4], shape0).round()
            pred[:-1, :, :4] = pred[:-1, :, :4] + bias
        pred[..., :4] = xyxy2xywh_(pred[..., :4])
        pred = pred.view((1, -1, pred.shape[-1]))

        # t30 = time_sync()
        # print(t30 - t3)
        pred = non_max_suppression(pred, conf_thres, iou_thres, labels=[], multi_label=True,
                                   agnostic=single_cls, nms_type=nms_type)
        # t31 = time_sync()
        # print(t31 - t30)
        dt[2] += time_sync() - t3

        labels[:, 1:] = xywhn2xyxy(labels[:, 1:], iw, ih)
        labels = torch.from_numpy(labels).cuda()

        path = Path(paths)
        if vis and pred[0].shape[0] > 0:
            viser.pred_vis(ori_img.copy(), pred[0], img_name=str(path.parts[-1]))

        # Metrics
        for si, pred_ in enumerate(pred):
            nl = len(labels)

            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred_) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred_[:, 5] = 0
            predn = pred_.clone()

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, ori_shapes[0], file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary

            if nl:
                correct = process_batch(predn, labels, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labels)
            else:
                correct = torch.zeros(pred_.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred_[:, 4].cpu(), pred_[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
            callbacks.run('on_val_image_end', predn, predn, path, names, im[si])

        if with_ori and plot_label and batch < plot_batch * plot_num:
            ratio, pad = ori_shapes[1], ori_shapes[2]

            target = labels.clone()
            target[:, 1:] = xyxy2xywhn(target[:, 1:], w=iw, h=ih)
            target[:, 1:] = xywhn2xyxy(target[:, 1:], ratio[0] * iw, ratio[1] * ih, padw=pad[0], padh=pad[1])
            target[:, 1:] = xyxy2xywh_(target[:, 1:])
            batch_ = batch - (batch // plot_num) * plot_num
            batch_ = batch_ * torch.ones((target.shape[0], 1), dtype=labels.dtype, device=labels.device)
            target = torch.cat((batch_, target), dim=1)
            targets_.append(target)

            pred_ = pred[0].clone()
            pred_[:, :4] = xyxy2xywhn(pred_[:, :4], w=iw, h=ih)
            pred_[:, :4] = xywhn2xyxy(pred_[:, :4], ratio[0] * iw, ratio[1] * ih, padw=pad[0], padh=pad[1])
            out.append(pred_)
            ims.append(im[-1])
            paths_.append(paths)

            if (batch + 1) % plot_num == 0:
                ims = torch.stack(ims, dim=0)
                targets_ = torch.cat(targets_, dim=0)
                f = save_dir / f'val_batch{(batch + 1) // plot_num}_labels.jpg'  # labels
                Thread(target=plot_images, args=(ims, targets_, paths_, f, names), daemon=True).start()
                f = save_dir / f'val_batch{(batch + 1) // plot_num}_pred.jpg'  # predictions
                Thread(target=plot_images, args=(ims, output_to_target(out), paths_, f, names), daemon=True).start()

                ims = []
                targets_ = []
                out = []
                paths_ = []
                if batch >= plot_batch * plot_num:
                    plot_label = False

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class_modified(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info('                Class       Images    Labels      P          R       mAP@.5    mAP@.5:.95')
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    # return (mp, mr, map50, map), maps, t

    # Save JSON
    if save_json and len(jdict):
        output_name, test_name, vis_name = make_name(opt)
        pred_json = str(save_dir / (vis_name + ".json"))  # predictions json
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)
    return (mp, mr, map50, map, *(loss.cpu() / num_loss).tolist()), maps, t


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_sub', type=str, default='val')
    parser.add_argument('--with_ori', type=str2bool, default=False)
    parser.add_argument('--nms_type', type=str, default='default', help='nms type')
    parser.add_argument('--post_cls', type=str2bool, default=False, help='nms type')

    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')

    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--augment', type=str2bool, default=False, help='augmented inference')
    parser.add_argument('--half', type=str2bool, default=False, help='use FP16 half-precision inference')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--save-txt', type=bool, default=False, help='save results to *.txt')
    parser.add_argument('--save-conf', type=bool, default=False, help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', type=bool, default=False, help='save a COCO-JSON results file')
    parser.add_argument('--vis', type=bool, default=False, help='save confidences in --save-txt labels')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')

    parser.add_argument('--single-cls', type=str2bool, default=False, help='treat as single-class dataset')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--task', default='study', help='train, val, test, speed or study')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    return opt


if __name__ == "__main__":
    opt = parse_opt()

    # opt.data = ROOT / 'data/VisDrone.yaml'
    # opt.val_sub = 'VisDrone2019-DET-train-test'
    # opt.val_sub = 'VisDrone2019-DET-test-dev'

    # opt.imgsz = 640  # 960
    # opt.weights = 'weights/best.pt'
    # # opt.conf_thres = 0.0001142
    # # opt.iou_thres = 0.3775415

    # opt.device = 1
    # opt.with_ori = True
    # # opt.nms_type = 'merge'

    # opt.vis = True
    opt.save_conf = True
    # opt.save_txt = True
    opt.save_json = True
    opt.batch_size = 1
    opt.half = True
    print_args(FILE.stem, opt)

    main(opt)






