from numpy import indices
import torch

def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y

def split_data(x, y, train_ratio=.8):
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = len(x) - train_cnt

    # Suffle dataset to split into train/valid set
    indices = torch.randperm(x.size(0))

    x = torch.index_select(
        x, 
        dim=0, 
        index=indices
        ).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(
        y, 
        dim=0, 
        index=indices
        ).split([train_cnt, valid_cnt], dim=0)
    
    return x, y

# input_size와 output_size 사이를 n_layers 개수만큼 등차 수열로 만들어 list로 반환
def get_hidden_size(input_size, output_size, n_layers):
    step_size = int((input_size - output_size) / n_layers)

    hidden_sizes = []
    current_size = input_size
    for i in range(n_layers-1):
        hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]

    return hidden_sizes



