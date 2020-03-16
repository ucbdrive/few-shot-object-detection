import argparse
import os
import yaml
from ast import literal_eval as make_tuple
from subprocess import Popen, PIPE, STDOUT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 3, 5, 10],
                        help='Shots to run experiments over')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 20],
                        help='Range of seeds to run')
    parser.add_argument('--root', type=str, default='./', help='Root of data')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of path')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--ckpt-freq', type=int, default=10,
                        help='Frequency of saving checkpoints')
    # Model
    parser.add_argument('--fc', action='store_true',
                        help='Model uses FC instead of cosine')
    parser.add_argument('--two-stage', action='store_true',
                        help='Two-stage fine-tuning')
    parser.add_argument('--novel-finetune', action='store_true',
                        help='Fine-tune novel weights first')
    parser.add_argument('--unfreeze', action='store_true',
                        help='Unfreeze feature extractor')
    # PASCAL arguments
    parser.add_argument('--split', '-s', type=int, default=1, help='Data split')
    # COCO arguments
    parser.add_argument('--coco', action='store_true', help='Use COCO dataset')

    args = parser.parse_args()
    return args


def load_yaml_file(fname):
    with open(fname, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_cmd(cmd):
    p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    while True:
        line = p.stdout.readline().decode('utf-8')
        if not line:
            break
        print(line)


def run_exp(cfg, configs):
    """
    Run training and evaluation scripts based on given config files.
    """
    # Train
    output_dir = configs['OUTPUT_DIR']
    model_path = os.path.join(args.root, output_dir, 'model_final.pth')
    if not os.path.exists(model_path):
        train_cmd = 'python tools/train_net.py --dist-url auto --num-gpus {} ' \
                    '--config-file {} --resume'.format(args.num_gpus, cfg)
        run_cmd(train_cmd)

    # Test
    res_path = os.path.join(args.root, output_dir, 'inference',
                            'res_final.json')
    if not os.path.exists(res_path):
        test_cmd = 'python tools/test_net.py --dist-url auto --num-gpus {} ' \
                   '--config-file {} --resume --eval-only'.format(args.num_gpus,
                                                                  cfg)
        run_cmd(test_cmd)


def get_config(seed, shot):
    """
    For a given seed and shot, generate a config file based on a template
    config file that is used for training/evaluation.
    You can extend/modify this function to fit your use-case.
    """
    if args.coco:
        # COCO
        assert args.two_stage, 'Only supports novel weights for COCO now'

        if args.novel_finetune:
            # Fine-tune novel classifier
            ITERS = {
                1: (10000, 500),
                2: (10000, 1500),
                3: (10000, 1500),
                5: (10000, 1500),
                10: (10000, 2000),
                30: (10000, 6000),
            }
            mode = 'novel'

            assert not args.fc and not args.unfreeze
        else:
            # Fine-tune entire classifier
            ITERS = {
                1: (14400, 16000),
                2: (28800, 32000),
                3: (43200, 48000),
                5: (72000, 80000),
                10: (144000, 160000),
                30: (216000, 240000),
            }
            mode = 'all'
        split = temp_split = ''
        temp_mode = mode

        config_dir = 'configs/COCO-detection'
        ckpt_dir = 'checkpoints/coco/faster_rcnn'
        base_cfg = '../../Base-RCNN-FPN.yaml'
    else:
        # PASCAL VOC
        assert not args.two_stage, 'Only supports random weights for PASCAL now'

        ITERS = {
            1: (3500, 4000),
            2: (7000, 8000),
            3: (10500, 12000),
            5: (17500, 20000),
            10: (35000, 40000),
        }
        split = 'split{}'.format(args.split)
        mode = 'all{}'.format(args.split)
        temp_split = 'split1'
        temp_mode = 'all1'

        config_dir = 'configs/PascalVOC-detection'
        ckpt_dir = 'checkpoints/voc/faster_rcnn'
        base_cfg = '../../../Base-RCNN-FPN.yaml'

    seed_str = 'seed{}'.format(seed) if seed != 0 else ''
    fc = '_fc' if args.fc else ''
    unfreeze = '_unfreeze' if args.unfreeze else ''
    # Read an example config file for the config parameters
    temp = os.path.join(
        temp_split, 'faster_rcnn_R_101_FPN_ft{}_{}_1shot{}'.format(
            fc, temp_mode, unfreeze)
    )
    config = os.path.join(args.root, config_dir, temp + '.yaml')

    prefix = 'faster_rcnn_R_101_FPN_ft{}_{}_{}shot{}{}'.format(
        fc, mode, shot, unfreeze, args.suffix)

    output_dir = os.path.join(args.root, ckpt_dir, seed_str)
    os.makedirs(output_dir, exist_ok=True)
    save_dir = os.path.join(
        args.root, config_dir, split, seed_str,
    )
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, prefix + '.yaml')

    configs = load_yaml_file(config)
    configs['_BASE_'] = base_cfg
    configs['DATASETS']['TRAIN'] = make_tuple(configs['DATASETS']['TRAIN'])
    configs['DATASETS']['TEST'] = make_tuple(configs['DATASETS']['TEST'])
    if args.coco and not args.novel_finetune:
        ckpt_path = os.path.join(output_dir, prefix, 'model_reset_combine.pth')
        if not os.path.exists(ckpt_path):
            src2 = os.path.join(
                output_dir, 'faster_rcnn_R_101_FPN_ft_novel_{}shot{}'.format(
                    shot, args.suffix),
                'model_final.pth',
            )
            if not os.path.exists(src2):
                print('Novel weights do not exist. Please run with the ' + \
                      '--novel-finetune flag first.')
                assert False
            combine_cmd = 'python tools/ckpt_surgery.py --coco --method ' + \
                'combine --src1 checkpoints/coco/faster_rcnn/faster_rcnn' + \
                '_R_101_FPN_base/model_final.pth --src2 {}'.format(src2) + \
                ' --save-dir {}'.format(os.path.join(output_dir, prefix))
            run_cmd(combine_cmd)
            assert os.path.exists(ckpt_path)
        configs['MODEL']['WEIGHTS'] = ckpt_path
    elif not args.coco:
        configs['MODEL']['WEIGHTS'] = configs['MODEL']['WEIGHTS'].replace(
            'base1', 'base' + str(args.split))
        for dset in ['TRAIN', 'TEST']:
            configs['DATASETS'][dset] = (
                configs['DATASETS'][dset][0].replace(
                    temp_mode, 'all' + str(args.split)),
            )
    configs['DATASETS']['TRAIN'] = (
        configs['DATASETS']['TRAIN'][0].replace(
            '1shot', str(shot) + 'shot'
        ) + ('_{}'.format(seed_str) if seed_str != '' else ''),
    )
    configs['SOLVER']['BASE_LR'] = args.lr
    configs['SOLVER']['MAX_ITER'] = ITERS[shot][1]
    configs['SOLVER']['STEPS'] = (ITERS[shot][0],)
    configs['SOLVER']['CHECKPOINT_PERIOD'] = ITERS[shot][1] // args.ckpt_freq
    configs['OUTPUT_DIR'] = os.path.join(output_dir, prefix)

    if seed != 0:
        with open(save_file, 'w') as fp:
            yaml.dump(configs, fp)

    return save_file, configs


def main(args):
    for shot in args.shots:
        for seed in range(args.seeds[0], args.seeds[1]):
            print('Split: {}, Seed: {}, Shot: {}'.format(args.split, seed, shot))
            cfg, configs = get_config(seed, shot)
            run_exp(cfg, configs)


if __name__ == '__main__':
    args = parse_args()
    main(args)
