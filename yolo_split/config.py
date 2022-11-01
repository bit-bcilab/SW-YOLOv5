

import albumentations as A

MODES = ['Visdrone', 'UAVDT']

PATCH_SETTINGS = {'Visdrone': [0.4, 0.4, 3, 3],
                  'UAVDT': [0.4, 0.4, 3, 3]}

SCALE_SETTINGS = {'Visdrone': [0.0, 0.0, 0.65, 0.65],
                  'UAVDT': [0.0, 0.0, 0.65, 0.65]}

SCALE_SETTINGS_VAL = {'Visdrone': [0.08, 0.08, 0.9, 0.9],
                      'UAVDT': [0.08, 0.08, 0.9, 0.9]}

ALBUMENT = {
    'Visdrone': A.Compose([
        A.Blur(p=0.3),
        A.MedianBlur(p=0.3),
        A.ToGray(p=0.08),
        A.RandomBrightnessContrast(p=0.3),
        A.CLAHE(p=0.1),
        A.RandomGamma(p=0.3),
        A.ImageCompression(quality_lower=70, p=0.45)],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']), p=0.5),

    'UAVDT': A.Compose([
        A.Blur(p=0.3),
        A.MedianBlur(p=0.3),
        A.ToGray(p=0.08),
        A.RandomBrightnessContrast(p=0.3),
        A.CLAHE(p=0.1),
        A.RandomGamma(p=0.3),
        A.ImageCompression(quality_lower=70, p=0.45)],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']), p=0.5)
}

HYP = {
    'Visdrone': {'num_rate': 1, 'ori_prob': 0.25,
                 # crop相关 参数
                 'max_occ': 0.60, 'short_thresh': 0.63,
                 'protect_rate': (-1.0, 0.10), 'scale_prob': 0.75, 'scale_jitter': 1.3,
                 # mosaic相关参数
                 'mosaic_vertical_prob': 0.04, 'mosaic_horizontal_prob': 0.08, 'mosaic_embed_prob': 0.16,
                 'mosaic_diag_r_prob': 0.20, 'mosaic_diag_l_prob': 0.25, 'mosaic_3_prob': 0.35, 'mosaic_4_prob': 0.5,
                 'embed_rate': (0.30, 0.60),
                 # copy-paste相关参数
                 'copy_paste_prob': 0.1, 'easy_paste_prob': 0.4, 'similar_paste_prob': 0.2, 'cluster_paste_prob': 0.3,
                 # 其他扩增参数
                 'perspective_prob': 0.5, 'hsv_prob': 0.4},

    'UAVDT': {'num_rate': 0.1, 'ori_prob': 0.25,
              # crop相关 参数
              'max_occ': 0.60, 'short_thresh': 0.63,
              'protect_rate': (-1.0, 0.10), 'scale_prob': 0.75, 'scale_jitter': 1.3,
              # mosaic相关参数
              'mosaic_vertical_prob': 0.04, 'mosaic_horizontal_prob': 0.08, 'mosaic_embed_prob': 0.16,
              'mosaic_diag_r_prob': 0.20, 'mosaic_diag_l_prob': 0.25, 'mosaic_3_prob': 0.35, 'mosaic_4_prob': 0.5,
              'embed_rate': (0.30, 0.60),
              # copy-paste相关参数
              'copy_paste_prob': 0.0, 'easy_paste_prob': 0.4, 'similar_paste_prob': 0.2, 'cluster_paste_prob': 0.3,
              # 其他扩增参数
              'perspective_prob': 0.5, 'hsv_prob': 0.4}
}


def set_attrs(obj, attrs):
    attr_names = list(attrs.keys())
    num_attrs = len(attr_names)
    for i in range(num_attrs):
        attr_name = attr_names[i]
        attr = attrs[attr_name]
        setattr(obj, attr_name, attr)
