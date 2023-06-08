import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from classification.data_loader import get_loaders
from classification.trainer import Trainer
from classification.model_loader import get_model


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.6)
    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--test_ratio', type=float, default=.2)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--verbose', type=int, default=2)
    # 모델 이름을 정해서 어떤 모델을 쓸건지 결정 (resnet, alexnet, vgg, squeezenet, densenet)
    p.add_argument('--model_name', type=str, default='resnet')
    # 데이터 셋은 어느것을 사용할건지
    p.add_argument('--dataset_name', type=str, default='catdog')
    # 데이터셋에 맞게 몇개의 클래스로 나눌건지 e.g.MNINST 같은 경우는 10개 (0~9)
    p.add_argument('--n_classes', type=int, default=2)

    # 1. 아무것도 적지 않음 : random init -> 거의 사용할일 없음
    # 2. use_pretrained 만 적음 : pretrained weights -> 때때로 필요할수있음
    # 3. freeze, use_pretrained 두가지 다 적음 : freeze pretrained weights -> 대부분 이것을 많이 사용
    # 4. freeze만 적으면 안된다
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--use_pretrained', action='store_true')

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    if config.verbose >= 2:
        print(config)

    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    model, input_size = get_model(config)
    model = model.to(device)

    train_loader, valid_loader, test_loader = get_loaders(config, input_size)

    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))
    print("Test:", len(test_loader.dataset))

    optimizer = optim.Adam(model.parameters())
    # logsoftmax이기 때문에 NLL사용
    crit = nn.NLLLoss()

    if config.verbose >= 2:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
