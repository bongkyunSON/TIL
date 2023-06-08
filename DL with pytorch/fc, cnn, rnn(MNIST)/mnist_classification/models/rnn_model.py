import torch
import torch.nn as nn


class SequenceClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,

        # 그레디언트 베니싱 문제를 해결했지만 time_step 기준으로만 해결한것이기에 4개이상은 권장하지 않음
        n_layers=4,
        dropout_p=.2,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,

            # LSTM은 batch_size defalt가 맨앞이 아니기 때문에 항상쓰는 batch_size가 앞으로 오게 하기위해서 True
            batch_first=True,

            # LSTM은 batchnorm을 사용할수 없다. 그러기에 dropout사용
            dropout=dropout_p,

            # MNIST같은 경우 한번에 데이터를 주는 형태이기 때문에 True
            bidirectional=True,
        )
        self.layers = nn.Sequential(
            nn.ReLU(),
            # bidirectional을 썼으니 hidden_size * 2 까먹지 말자
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # 사실상 h = time_step의 크기, w = 입력된 벡터의 크기
        # |x| = (batch_size, h, w)

        z, _ = self.rnn(x)
        # |z| = (batch_size, h, hidden_size * 2)

        z = z[:, -1]
        # 마지막 time_step만 받아오기 위해
        # |z| = (batch_size, hidden_size * 2)
        
        y = self.layers(z)
        # |y| = (batch_size, output_size)

        return y
