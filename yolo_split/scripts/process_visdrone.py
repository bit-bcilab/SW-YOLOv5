

from PIL import Image
from tqdm import tqdm
from glob import glob
from os.path import join
from multiprocessing import Pool
from utils.general import Path


def convert_box_visdrone(size, box):
    # Convert VisDrone box to YOLO xywh box
    dw = 1. / size[0]
    dh = 1. / size[1]
    return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh


def _visdrone2yolo(args):
    root = args[0]
    anno_name = args[1]
    filtering = args[2]
    anno_name = anno_name.replace('\\', '/')
    img_size = Image.open((root / 'images' / anno_name.split('/')[-1]).with_suffix('.jpg')).size
    lines = []
    with open(anno_name, 'r') as file:  # 读取 read annotation.txt
        for row in [x.split(',') for x in file.read().strip().splitlines()]:
            cls = int(row[5]) - 1
            box = convert_box_visdrone(img_size, tuple(map(int, row[:4])))

            """
            set conditions for filtering 设置条件来筛选标签
            """
            # 筛掉忽略区域
            if row[4] == '0':  # VisDrone 'ignored regions' class 0
                continue

            # 在训练集中筛掉严重遮挡/出视野的
            if filtering and (int(row[6]) > 1 or int(row[7]) > 1):
                continue

            # 筛掉非汽车/非载具
            # if cls != 3 and cls != 4 and cls != 5 and cls != 8:
            #     continue
            # cls = 0

            # 筛掉尺寸过大的
            # if not (box[2] * box[3] < (0.06 * 0.06)):
            #     continue

            # 筛掉异常尺寸
            if filtering and (box[0] < 0. or box[1] < 0. or box[2] <= 0. or box[3] <= 0.):
                continue

            lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
            # with open(str(anno_name).replace(os.sep + 'annotations' + os.sep, os.sep + 'labels' + os.sep), 'w') as fl:
            #     fl.writelines(lines)  # write label.txt
            with open(str(anno_name).replace('annotations', 'labels'), 'w') as fl:
                fl.writelines(lines)  # write label.txt
        file.close()
        return 0


def visdrone2yolo(root, filtering=True, num_process=8):
    (root / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory

    annos = glob(join((root / 'annotations'), '*.txt'))
    num_annos = len(annos)

    # for n in range(num_annos):
    #     _visdrone2yolo([root, annos[n]])
    with Pool(processes=num_process) as pool:
        for n in tqdm(pool.imap_unordered(_visdrone2yolo, zip([root] * num_annos, annos, [filtering] * num_annos)),
                      desc='process annos', total=num_annos, ncols=100):
            pass


if __name__ == "__main__":
    root = Path('/home/Data/Visdrone2019-DET/')
    filtering = True
    for d in ['VisDrone2019-DET-train-test', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']:
        visdrone2yolo(root / d, filtering)  # convert VisDrone annotations to YOLO labels
