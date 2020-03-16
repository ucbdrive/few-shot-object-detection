import argparse
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shots', type=int, default=1,
                        help='Shots to aggregate over')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Seeds to aggregate over')
    # Model
    parser.add_argument('--fc', action='store_true',
                        help='Model uses FC instead of cosine')
    parser.add_argument('--unfreeze', action='store_true',
                        help='Unfreeze feature extractor')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of path')
    # Output arguments
    parser.add_argument('--print', action='store_true', help='Clean output')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    parser.add_argument('--save-dir', type=str, default='.',
                        help='Save dir for generated plots')
    # PASCAL arguments
    parser.add_argument('--split', type=int, default=1, help='Data split')
    # COCO arguments
    parser.add_argument('--coco', action='store_true', help='Use COCO dataset')

    args = parser.parse_args()
    return args


def main(args):
    metrics = {}
    num_ckpts = 0
    dataset = 'coco' if args.coco else 'voc'
    if args.fc:
        fc = '_fc'
    else:
        fc = '_normalized' if not args.coco else ''
    if args.unfreeze:
        unfreeze = '_unfreeze'
    else:
        unfreeze = '_randnovel' if not args.coco else ''
    for i in range(args.seeds):
        seed = 'seed{}/'.format(i) if i != 0 else ''
        prefix = 'checkpoints/{}/faster_rcnn/{}'.format(dataset, seed)
        prefix += 'faster_rcnn_R_101_FPN_ft{}_all'.format(fc)
        if args.coco:
            ckpt = prefix + '_{}shot{}'.format(args.shots, unfreeze)
        else:
            ckpt = prefix + '{}_{}shot{}{}'.format(
                args.split, args.shots, unfreeze, args.suffix)
        if os.path.exists(ckpt):
            if os.path.exists(os.path.join(ckpt, 'inference/all_res.json')):
                ckpt_ = os.path.join(ckpt, 'inference/all_res.json')
                res = json.load(open(ckpt_, 'r'))
                res = res[os.path.join(ckpt, 'model_final.pth')]['bbox']
            elif os.path.exists(os.path.join(ckpt, 'inference/res_final.json')):
                ckpt = os.path.join(ckpt, 'inference/res_final.json')
                res = json.load(open(ckpt, 'r'))['bbox']
            else:
                print('Missing: {}'.format(ckpt))
                continue
            
            for metric in res:
                if metric in metrics:
                    metrics[metric].append(res[metric])
                else:
                    metrics[metric] = [res[metric]]
            num_ckpts += 1
        else:
            print('Missing: {}'.format(ckpt))
    print('Num ckpts: {}'.format(num_ckpts))
    print('')

    # Output results
    if args.print:
        # Clean output for copy and pasting
        out_str = ''
        for metric in metrics:
            out_str += '{0:.1f} '.format(np.mean(metrics[metric]))
        print(out_str)
        out_str = ''
        for metric in metrics:
            out_str += '{0:.1f} '.format(
                1.96*np.std(metrics[metric]) / math.sqrt(len(metrics[metric]))
            )
        print(out_str)
        out_str = ''
        for metric in metrics:
            out_str += '{0:.1f} '.format(np.std(metrics[metric]))
        print(out_str)
        out_str = ''
        for metric in metrics:
            out_str += '{0:.1f} '.format(np.percentile(metrics[metric], 25))
        print(out_str)
        out_str = ''
        for metric in metrics:
            out_str += '{0:.1f} '.format(np.percentile(metrics[metric], 50))
        print(out_str)
        out_str = ''
        for metric in metrics:
            out_str += '{0:.1f} '.format(np.percentile(metrics[metric], 75))
        print(out_str)
    else:
        # Verbose output
        for metric in metrics:
            print(metric)
            print('Mean \t {0:.4f}'.format(np.mean(metrics[metric])))
            print('Std \t {0:.4f}'.format(np.std(metrics[metric])))
            print('Q1 \t {0:.4f}'.format(np.percentile(metrics[metric], 25)))
            print('Median \t {0:.4f}'.format(np.percentile(metrics[metric], 50)))
            print('Q3 \t {0:.4f}'.format(np.percentile(metrics[metric], 75)))
            print('')

    # Plot results
    if args.plot:
        os.makedirs(args.save_dir, exist_ok=True)

        for met in ['avg', 'stdev', 'ci']:
            for metric, c in zip(['nAP', 'nAP50', 'nAP75'],
                                 ['bo-', 'ro-', 'go-']):
                if met == 'avg':
                    res = [np.mean(metrics[metric][:i+1]) \
                            for i in range(len(metrics[metric]))]
                elif met == 'stdev':
                    res = [np.std(metrics[metric][:i]) \
                            for i in range(1, len(metrics[metric])+1)]
                elif met == 'ci':
                    res = [1.96*np.std(metrics[metric][:i+1]) / \
                            math.sqrt(len(metrics[metric][:i+1])) \
                                for i in range(len(metrics[metric]))]
                plt.plot(range(1, len(metrics[metric])+1), res, c)
            plt.legend(['nAP', 'nAP50', 'nAP75'])
            plt.title('Split {}, {} Shots - Cumulative {} over {} Seeds'.format(
                args.split, args.shots, met.upper(), args.seeds))
            plt.xlabel('Number of seeds')
            plt.ylabel('Cumulative {}'.format(met.upper()))
            plt.savefig(os.path.join(
                args.save_dir,
                'split{}_{}shots_{}_vs_{}seeds.png'.format(
                    args.split, args.shots, met, args.seeds),
            ))
            plt.clf()


if __name__ == '__main__':
    args = parse_args()
    main(args)
