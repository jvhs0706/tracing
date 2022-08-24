import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse

from models import *
from datasets import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help = 'Benchmark model structure.', choices = ['VGG16', 'ResNet18'])
    parser.add_argument('dataset_name', help = 'Benchmark dataset used.', choices = ['CIFAR10', 'GTSRB'])
    parser.add_argument('-b', '--batch_size', help = 'Batch size.', type = int, default = 128)
    parser.add_argument('-e', '--num_epochs', help = 'Number of epochs.', type = int, default = 50)
    parser.add_argument('-lr', '--learning_rate', help = 'Learning rate.', type = float, default = 1e-3)
    args = parser.parse_args()

    # renaming
    training_set, testing_set = eval(f'{args.dataset_name}_training_set'), eval(f'{args.dataset_name}_testing_set')
    num_classes = eval(f'{args.dataset_name}_num_classes')
    means, stds = eval(f'{args.dataset_name}_means'), eval(f'{args.dataset_name}_stds')
    Head, Tail = eval(f'{args.model_name}Head'), eval(f'{args.model_name}Tail')

    # place to save the trained model
    save_dir = f'saved_models/{args.model_name}-{args.dataset_name}'
    os.makedirs(save_dir, exist_ok = True)

    # set up the model
    base_model = nn.Sequential(transforms.Normalize(means, stds), Head(), Tail(num_classes))
    base_model.cuda()

    print(f'The head has {sum(p.numel() for p in base_model[1].parameters())} parameters, the tail has {sum(p.numel() for p in base_model[2].parameters())} parameters.')

    # prepare for training
    optimizer = torch.optim.Adam(base_model.parameters(), lr = args.learning_rate)
    Loss = nn.CrossEntropyLoss()
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle = True, num_workers = 2)
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=args.batch_size, num_workers = 2)

    # training loop
    best_accuracy = 0.0
    for n in range(args.num_epochs):
        # training
        base_model.train()
        epoch_loss = 0.0
        for X, y in training_loader:
            optimizer.zero_grad()
            loss = Loss(base_model(X.cuda()), y.cuda())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y) / len(training_set)

        # testing
        base_model.eval()
        accuracy = 0.0
        with torch.no_grad():
            for X, y in testing_loader:
                _, pred = base_model(X.cuda()).max(axis = -1)
                accuracy += (pred == y.cuda()).sum().item() / len(testing_set)

        print(f'Epoch {n}, loss {epoch_loss:.3f}, accuracy = {accuracy:.4f}.')

        # save when the accuracy is good
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(base_model[1].state_dict(), f'{save_dir}/base_head_state_dict')
            torch.save(base_model[2].state_dict(), f'{save_dir}/base_tail_state_dict')
        
    print(f'Completed the training of the base model, {args.model_name}-{args.dataset_name}, accuracy = {best_accuracy:.4f}.')