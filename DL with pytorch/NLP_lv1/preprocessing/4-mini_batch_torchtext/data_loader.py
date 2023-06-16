from torchtext import data


class DataLoader(object):
    '''
    Data loader class to load text file using torchtext library.
    '''

    def __init__(
        self, train_fn,
        batch_size=64,
        valid_ratio=.2,
        device=-1,
        max_vocab=999999,
        min_freq=1,
        use_eos=False,
        shuffle=True
    ):
        '''
        DataLoader initialization.
        :param train_fn: Train-set filename
        :param batch_size: Batchify data fot certain batch size.
        :param device: Device-id to load data (-1 for CPU)
        :param max_vocab: Maximum vocabulary size
        :param min_freq: Minimum frequency for loaded word.
        :param use_eos: If it is True, put <EOS> after every end of sentence.
        :param shuffle: If it is True, random shuffle the input data.
        '''
        super().__init__()

        # Define field of the input file.
        # The input file consists of two fields.
        # 라벨은 e.g. positive, nagative 같은 라벨링
        self.label = data.Field(
            # 시퀀셜 데이터가 아니기 때문에 False
            sequential=False,
            # vocab은 True가 좋다 몇개인지 알면 좋다
            use_vocab=True,
            # unknown은 있으면 안된다
            unk_token=None
        )
        # 말그대로 라벨 말고 텍스트 e.g. 댓글같은것들
        self.text = data.Field(
            # vocab은 당연히 True
            use_vocab=True,
            # batch first는 알지?
            batch_first=True,
            # 현재는 필요없음 나중에 쓰일거임
            include_lengths=False,
            # 이것도 지금은 불필요
            eos_token='<EOS>' if use_eos else None
        )

        # Those defined two columns will be delimited by TAB.
        # Thus, we use TabularDataset to load two columns in the input file.
        # We would have two separate input file: train_fn, valid_fn
        # Files consist of two columns: label field and text field.
        train, valid = data.TabularDataset(
            path=train_fn,
            format='tsv', 
            fields=[
                ('label', self.label),
                ('text', self.text),
            ],
        ).split(split_ratio=(1 - valid_ratio))

        # Those loaded dataset would be feeded into each iterator:
        # train iterator and valid iterator.
        # We sort input sentences by length, to group similar lengths.
        self.train_loader, self.valid_loader = data.BucketIterator.splits(
            (train, valid),
            batch_size=batch_size,
            device='cuda:%d' % device if device >= 0 else 'cpu',
            shuffle=shuffle,
            # 비슷한 길이의 문장 끼리 mini_batch할때 쓰이는 key 
            sort_key=lambda x: len(x.text),
            # 미니배치 안에서 sort
            sort_within_batch=True,
        )

        # At last, we make a vocabulary for label and text field.
        # It is making mapping table between words and indice.
        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size=max_vocab, min_freq=min_freq)
