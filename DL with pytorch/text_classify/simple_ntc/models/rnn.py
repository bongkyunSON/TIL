import torch.nn as nn


class RNNClassifier(nn.Module):

    def __init__(
        self,
        # input size는 porch text가 정해줄거다
        input_size,
        word_vec_size,
        hidden_size,
        n_classes,
        # LSTM은 layers를 4개 이상하면 안됨
        n_layers=4,
        dropout_p=.3,
    ):
        self.input_size = input_size  # vocabulary_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()
        # Embedding layer는 linear layear랑 수학적으로 똑같고 구현상으로 효율적으로 구현해 놓은것
        self.emb = nn.Embedding(input_size, word_vec_size)
        self.rnn = nn.LSTM(
            input_size=word_vec_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            # LSTM은 batchnorm을 사용할수 없다. 그러기에 dropout사용
            dropout=dropout_p,
            # batch_first는 batch size를 1번으로, 안해주면 batch_size가 2번째로 간다
            batch_first=True,
            # 한번에 입력이 들어올때는 True, 타임스탭 별로 들어오면 False
            bidirectional=True,
        )
        # bidirectional 이기때문에 hidden size *2 ->many to one은 bidirectional
        self.generator = nn.Linear(hidden_size * 2, n_classes)
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_size)
        # LSTM 같은 경우 출력이 2가지. 1번이 output, 2번이 마지막 timestep에 hidden state과 sell state
        x, _ = self.rnn(x)
        # |x| = (batch_size, length, hidden_size * 2)
        # many to one이기 때문에 마지막만 슬라이싱
        y = self.activation(self.generator(x[:, -1]))
        # |y| = (batch_size, n_classes)

        return y
