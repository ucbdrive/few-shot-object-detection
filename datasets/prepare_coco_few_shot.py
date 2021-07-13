import argparse
import json
import os
import random
import sys

from fsdet.utils import io

PROJ_ROOT = str(io.get_project_root())
# TODO: This has to be injected either as an environment variable or a parameter.
DATASET_ROOT = os.path.join(PROJ_ROOT, 'datasets/socket_plates')
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 6],
                        help="Range of seeds")
    args = parser.parse_args()
    return args


def generate_seeds(args):
    data_path = os.path.join(ANN_ROOT, 'train.json')
    data = json.load(open(data_path))

    new_all_cats = []
    for cat in data['categories']:
        new_all_cats.append(cat)

    id2img = {}
    for i in data['images']:
        id2img[i['id']] = i

    anno = {i: [] for i in ID2CLASS.keys()}
    for a in data['annotations']:
        anno[a['category_id']].append(a)

    for i in range(args.seeds[0], args.seeds[1]):
        random.seed(i)
        for c in [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            img_ids = {}
            if not anno[c]:
                continue
            for a in anno[c]:
                if a['image_id'] in img_ids:
                    img_ids[a['image_id']].append(a)
                else:
                    img_ids[a['image_id']] = [a]
            sample_shots = []
            sample_imgs = []
            for shots in [1, 2, 3, 5, 10]:
                while True:
                    imgs = random.sample(list(img_ids.keys()), shots)
                    for img in imgs:
                        skip = False
                        for s in sample_shots:
                            if img == s['image_id']:
                                skip = True
                                break
                        if skip:
                            continue
                        if len(img_ids[img]) + len(sample_shots) > shots:
                            continue
                        sample_shots.extend(img_ids[img])
                        sample_imgs.append(id2img[img])
                        if len(sample_shots) == shots:
                            break
                    if len(sample_shots) == shots:
                        break
                new_data = {
                    #'info': data['info'],
                    #'licenses': data['licenses'],
                    'images': sample_imgs,
                    'annotations': sample_shots,
                }
                save_path = get_save_path_seeds(data_path, ID2CLASS[c], shots, i)
                new_data['categories'] = new_all_cats
                with open(save_path, 'w') as f:
                    json.dump(new_data, f)


def get_save_path_seeds(path, cls, shots, seed):
    s = path.split('/')
    prefix = 'full_box_{}shot_{}_train'.format(shots, cls)
    save_dir = os.path.join(DATASET_ROOT, 'seed' + str(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + '.json')
    return save_path


if __name__ == '__main__':
    ID2CLASS = {
        1: 'AOP_EVK80',
        2: 'AOP_TRAS1000',
        3: 'AOP_TRAS1000_no_key',
        #4: 'AOP_X10DER_KT_01',
        5: 'SPLITTER_MCP_03',
        6: 'SPLITTER_POA_01_met_kapje',
        7: 'SPLITTER_POA_01_zonder_kapje',
        8: 'SPLITTER_POA_01IEC',
        9: 'SPLITTER_POA_3_met_kapje',
        10: 'SPLITTER_POA_3_zonder_kapje',
        11: 'SPLITTER_SQ601_met_kapje',
        12: 'SPLITTER_UMU_met_kapje',
        13: 'WCD_tweegats',
        14: 'AOP_BTV1',
        15: 'AOP_DIO_01',
    }
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}

    args = parse_args()
    generate_seeds(args)
    sys.exit(0)
