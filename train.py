import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer

from utils import load_mnist
from utils import split_data
from utils import get_hideen_size

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    
    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)

    p.add_argument('--n_layers', type=int, default=5)
    p.add_argument('--use_dropout', action='store_true')
    p.add_argument('--dropout_p', type=float, default=.3)

    p.add_argument('--verbose', type=int, default=1)

    config = p.parse_args()

    return config

def main(config):
    # Set device based on user defined configuaration
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    x, y = load_mnist(is_train=True, flatten=True)
    x, y = split_data(x.to(device), y.to(device), train_raito=config.train_ratio)

    print('Train:', x[0].shape, y[0].shape)
    print('Valid:', x[1].shape, y[1].shape)

    # flatten = True 일때만 작업 가능
    input_size = int(x[0].shape[-1])
    # MNIST에만 적용되는 output_size
    output_size = int(max(y[0])) + 1

    model = ImageClassifier(
        x
    )






if __name__ == '__main__':
    config = define_argparser()
    main(config)

