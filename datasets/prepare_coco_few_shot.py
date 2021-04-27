import argparse
import json
import os
import random
<<<<<<< HEAD


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 10],
=======
import sys

PROJ_ROOT = '/home/irene/few-shot-object-detection/'
DATASET_ROOT = os.path.join(PROJ_ROOT, 'datasets')
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 6],
>>>>>>> Remove binaries, environment and the datasets as those are not supposed to be on GitHub.
                        help="Range of seeds")
    args = parser.parse_args()
    return args

<<<<<<< HEAD

def generate_seeds(args):
    data_path = 'datasets/cocosplit/datasplit/trainvalno5k.json'
=======
def generate_seeds(args):
    data_path = os.path.join(ANN_ROOT, 'train.json')
>>>>>>> Remove binaries, environment and the datasets as those are not supposed to be on GitHub.
    data = json.load(open(data_path))

    new_all_cats = []
    for cat in data['categories']:
        new_all_cats.append(cat)

    id2img = {}
    for i in data['images']:
        id2img[i['id']] = i

    anno = {i: [] for i in ID2CLASS.keys()}
    for a in data['annotations']:
<<<<<<< HEAD
        if a['iscrowd'] == 1:
            continue
=======
>>>>>>> Remove binaries, environment and the datasets as those are not supposed to be on GitHub.
        anno[a['category_id']].append(a)

    for i in range(args.seeds[0], args.seeds[1]):
        random.seed(i)
<<<<<<< HEAD
        for c in ID2CLASS.keys():
            img_ids = {}
=======
        #category_ids that are not empty! if you include one or more without annotations use line 39 and 40:
        for c in [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            img_ids = {}
            if not anno[c]:
                continue
>>>>>>> Remove binaries, environment and the datasets as those are not supposed to be on GitHub.
            for a in anno[c]:
                if a['image_id'] in img_ids:
                    img_ids[a['image_id']].append(a)
                else:
                    img_ids[a['image_id']] = [a]
<<<<<<< HEAD

            sample_shots = []
            sample_imgs = []
            for shots in [1, 2, 3, 5, 10, 30]:
=======
            sample_shots = []
            sample_imgs = []
            for shots in [1, 2, 3, 5, 10]:
>>>>>>> Remove binaries, environment and the datasets as those are not supposed to be on GitHub.
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
<<<<<<< HEAD
                    'info': data['info'],
                    'licenses': data['licenses'],
=======
                    #'info': data['info'],
                    #'licenses': data['licenses'],
>>>>>>> Remove binaries, environment and the datasets as those are not supposed to be on GitHub.
                    'images': sample_imgs,
                    'annotations': sample_shots,
                }
                save_path = get_save_path_seeds(data_path, ID2CLASS[c], shots, i)
                new_data['categories'] = new_all_cats
                with open(save_path, 'w') as f:
                    json.dump(new_data, f)

<<<<<<< HEAD

def get_save_path_seeds(path, cls, shots, seed):
    s = path.split('/')
    prefix = 'full_box_{}shot_{}_trainval'.format(shots, cls)
    save_dir = os.path.join('datasets', 'cocosplit', 'seed' + str(seed))
=======
def get_save_path_seeds(path, cls, shots, seed):
    s = path.split('/')
    prefix = 'full_box_{}shot_{}_train'.format(shots, cls)
    save_dir = os.path.join(DATASET_ROOT, 'seed' + str(seed))
>>>>>>> Remove binaries, environment and the datasets as those are not supposed to be on GitHub.
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + '.json')
    return save_path

<<<<<<< HEAD

if __name__ == '__main__':
    ID2CLASS = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
=======
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
>>>>>>> Remove binaries, environment and the datasets as those are not supposed to be on GitHub.
    }
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}

    args = parse_args()
    generate_seeds(args)
<<<<<<< HEAD
=======
    sys.exit(0)
>>>>>>> Remove binaries, environment and the datasets as those are not supposed to be on GitHub.
