

import sys
from pycocotools.coco import COCO
from utils.coco_eval import COCOeval  # from pycocotools.cocoeval import COCOeval


def coco_eval(pred_path, anno_path):
    anno = COCO(anno_path)
    pred = anno.loadRes(pred_path)
    coco_eval = COCOeval(anno, pred, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    ap, ap50 = coco_eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)


if __name__ == "__main__":
    pred_path = 'tune_results/split-0.32×4-test1.json'
    # anno_path = 'D://DataBase/VisDrone2019-DET/Anno/VisDrone2019-DET_test-dev_coco.json'
    anno_path = 'data/VisDrone2019-DET_test-dev_coco.json'

    inputs = sys.argv
    if len(inputs) == 2:
        pred_path = inputs[1]
    elif len(inputs) == 3:
        pred_path, anno_path = inputs[1], inputs[2]

    coco_eval(pred_path, anno_path)
