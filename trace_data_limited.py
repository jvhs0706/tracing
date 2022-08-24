import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import argparse

from models import *
from datasets import *
from watermark import Watermark

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help = 'Benchmark model structure.', choices = ['VGG16', 'ResNet18'])
    parser.add_argument('dataset_name', help = 'Benchmark dataset used.', choices = ['CIFAR10', 'GTSRB'])
    parser.add_argument('attacks', help = 'Attacks to be explored.', nargs = '+')
    parser.add_argument('-M', '--num_models', help = 'The number of models used.', type = int, default = 100)
    parser.add_argument('-n', '--num_samples', help = 'The number of adversarial samples per model.', type = int, default = 10)
    args = parser.parse_args()

    # renaming
    training_set, testing_set = eval(f'{args.dataset_name}_training_set'), eval(f'{args.dataset_name}_testing_set')
    num_classes = eval(f'{args.dataset_name}_num_classes')
    means, stds = eval(f'{args.dataset_name}_means'), eval(f'{args.dataset_name}_stds')
    Head, Tail = eval(f'{args.model_name}Head'), eval(f'{args.model_name}Tail')

    # input and output directories
    model_dir = f'saved_models/{args.model_name}-{args.dataset_name}'
    adv_dir = f'saved_adv_examples/{args.model_name}-{args.dataset_name}-{args.num_models}heads'

    # load the tail of the model
    normalizer = transforms.Normalize(means, stds)
    tail = Tail(num_classes)
    tail.load_state_dict(torch.load(f'{model_dir}/base_tail_state_dict'))
    tail.cuda()
    tail.eval()

    # load the classifiers
    heads, watermarks = [], []
    for i in range(args.num_models):
        heads.append(Head())
        heads[-1].cuda()
        heads[-1].load_state_dict(torch.load(f'{model_dir}/head_{i}/state_dict'))
        heads[-1].eval()
        watermarks.append(Watermark.load(f'{model_dir}/head_{i}/watermark.npy'))
    
    for a in args.attacks:
        correct = 0
        for i in tqdm(range(args.num_models)):
            adv_npz = np.load(f'{adv_dir}/head_{i}/{a}.npz')
            X, X_attacked, y = adv_npz['X'][:args.num_samples], adv_npz['X_attacked'][:args.num_samples], adv_npz['y'][:args.num_samples]

            diffs_sum = np.stack([wm.get_values(np.abs(X - X_attacked)).sum(axis = -1) for wm in watermarks], axis = 1)

            with torch.no_grad():
                out = torch.stack([head(wm(normalizer(torch.tensor(X_attacked, device = 'cuda')))) for wm, head in zip(watermarks, heads)], axis = 1) # (args.num_samples, args.num_models, ...)
                out = tail(out.view(args.num_samples * args.num_models, *out.shape[2:])).view(args.num_samples, args.num_models, num_classes)
                wrong_pred = out.cpu().numpy().argmax(axis = -1) != y[:, None]
            diffs_sum[~wrong_pred] = np.inf
            correct += np.sum(diffs_sum.argmin(axis = 1) == i)

        print(f'Attack {a}, tracing accuracy {correct / (args.num_models * args.num_samples)}.')

