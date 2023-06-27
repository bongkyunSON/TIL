import sys
import argparse

import torch
import torch.nn as nn
# version = list(map(int, torchtext.__version__.split('.')))
# if version[0] <= 0 and version[1] < 9:
#     from torchtext import data
# else:
#     from torchtext.legacy import data
from torchtext import data
from simple_ntc.models.rnn import RNNClassifier
from simple_ntc.models.cnn import CNNClassifier


def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    # 확률이 높은 순서대로 출력해주는거 e.g. top_k = 3 이면 확율이 높은 순서대로 3가지를 출력해줌
    p.add_argument('--top_k', type=int, default=1)
    p.add_argument('--max_length', type=int, default=256)
    
    p.add_argument('--drop_rnn', action='store_true')
    p.add_argument('--drop_cnn', action='store_true')

    config = p.parse_args()

    return config


def read_text(max_length=256):
    '''
    Read text from standard input for inference.
    '''
    lines = []

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split(' ')[:max_length]]

    return lines


def define_field():
    '''
    To avoid use DataLoader class, just declare dummy fields. 
    With those fields, we can retore mapping table between words and indice.
    '''

    # data loader를 사용하지 않는다
    return (
        data.Field(
            use_vocab=True,
            batch_first=True,
            include_lengths=False,
        ),
        data.Field(
            sequential=False,
            use_vocab=True,
            unk_token=None,
        )
    )


def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    rnn_best = saved_data['rnn']
    cnn_best = saved_data['cnn']
    vocab = saved_data['vocab']
    classes = saved_data['classes']

    vocab_size = len(vocab)
    n_classes = len(classes)

    # 원핫이 들어오면 단어로 변환하거나 단어가 주어지면 원핫으로 변환하거나 
    # 클래스명이 주어지면 클래스 인덱스로 변환하고 클래스의 인덱스가 주어지면 클래스명으로 변환하거나 하는 맵핑함수
    text_field, label_field = define_field()
    text_field.vocab = vocab
    label_field.vocab = classes

    lines = read_text(max_length=config.max_length)

    # 학습 하는게 아니기 때문에 no_grad 잊으면 안됨!
    with torch.no_grad():
        ensemble = []
        if rnn_best is not None and not config.drop_rnn:
            # Declare model and load pre-trained weights.
            model = RNNClassifier(
                input_size=vocab_size,
                word_vec_size=train_config.word_vec_size,
                hidden_size=train_config.hidden_size,
                n_classes=n_classes,
                n_layers=train_config.n_layers,
                dropout_p=train_config.dropout,
            )
            model.load_state_dict(rnn_best)
            ensemble += [model]
        if cnn_best is not None and not config.drop_cnn:
            # Declare model and load pre-trained weights.
            model = CNNClassifier(
                input_size=vocab_size,
                word_vec_size=train_config.word_vec_size,
                n_classes=n_classes,
                use_batch_norm=train_config.use_batch_norm,
                dropout_p=train_config.dropout,
                window_sizes=train_config.window_sizes,
                n_filters=train_config.n_filters,
            )
            model.load_state_dict(cnn_best)
            ensemble += [model]

        y_hats = []
        # Get prediction with iteration on ensemble.
        for model in ensemble:
            if config.gpu_id >= 0:
                model.cuda(config.gpu_id)
            # Don't forget turn-on evaluation mode.
            # eval 까먹으면 안돼!
            model.eval()

            y_hat = []
            for idx in range(0, len(lines), config.batch_size):                
                # Converts string to list of index.
                x = text_field.numericalize(
                    text_field.pad(lines[idx:idx + config.batch_size]),
                    device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu',
                )

                y_hat += [model(x).cpu()]
            # Concatenate the mini-batch wise result
            y_hat = torch.cat(y_hat, dim=0)
            # |y_hat| = (len(lines), n_classes)

            y_hats += [y_hat]

            model.cpu()
        # Merge to one tensor for ensemble result and make probability from log-prob.
        # log확률이 들어있기 때문에 exp()를해서 확률로 바꿈
        y_hats = torch.stack(y_hats).exp()

        # |y_hats| = (len(ensemble), len(lines), n_classes)
        # y_hats = y_hats.mean(dim=0) 이랑 같음
        y_hats = y_hats.sum(dim=0) / len(ensemble) # Get average
        # |y_hats| = (len(lines), n_classes)

        # top_k를 쓰면 probs와 indice로 나온다
        # probs는 실제 확률값이 Float tensor로 나오고
        # indice는 aregument_index가 longtensor로 나오고 
        probs, indice = y_hats.topk(config.top_k)

        for i in range(len(lines)):
            sys.stdout.write('%s\t%s\n' % (
                # itos를 사용하면 int를 str로 바꿔줌 -> 클래스 명이 나온다
                ' '.join([classes.itos[indice[i][j]] for j in range(config.top_k)]), 
                ' '.join(lines[i])
            ))


if __name__ == '__main__':
    config = define_argparser()
    main(config)
