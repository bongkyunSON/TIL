from copy import deepcopy

import numpy as np

import torch

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from simple_ntc.utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MyEngine(Engine):

    def __init__(self, func, model, crit, optimizer, config):
        # Ignite Engine does not have objects in below lines.
        # Thus, we assign class variables to access these object, during the procedure.
        # func은 ignite engine을 사용할때 init에서 받아줘야한다
        # model, crit, optimizer, config -> ignite engine이 들고있지 않아서 따로 상속을 해준다
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func) # Ignite Engine only needs function to run.
        # best loss만 기억하고있을 애들
        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod
    def train(engine, mini_batch):
        # mini_batch 안에는 튜플이다
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train() # Because we assign model as class variable, we can easily access to it.
        engine.optimizer.zero_grad()

        # text는 댓글의 long_tensor, label은 class에 long_tensor
        x, y = mini_batch.text, mini_batch.label
        # model에 디바이스와 같은 디바이스로 옮겨주기
        x, y = x.to(engine.device), y.to(engine.device)
        # 데이터가 쓸데없이 긴것들이 있어서 max_length로 짤라준다
        x = x[:, :engine.config.max_length]

        # Take feed-forward
        y_hat = engine.model(x)
        
        # crit는 NLLLoss
        # Loss계산
        loss = engine.crit(y_hat, y)
        # Backpropagation
        loss.backward()

        # Calculate accuracy only if 'y' is LongTensor,
        # which means that 'y' is one-hot representation.
        # y가 long tensor이면 accuracy를 구하는 코드
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0
        

        p_norm = float(get_parameter_norm(engine.model.parameters())) # model에 L2_norm의 합 -> 보통 학습이 진행되면 L2_norm이 커진다

        g_norm = float(get_grad_norm(engine.model.parameters())) # 그레디언트의 L2-norm -> 수치가 크면 파라미터가 조금만 바뀌어도 크게반응
                                                                # 작으면 작게반응, 이걸 통해서 loss surface가 얼마나 가파른지 알수있다
                                                                # 학습이 처음 시작할때 수치가 높고 가면갈수록 작아지면서 수렴하려고한다
                                                                # 학습이 불안정하면 값이 날뛰면서 NAN이 뜰 수 있다
                                                                # 즉 학습의 안전성을 알 수 있다.

        # Take a step of gradient descent.
        engine.optimizer.step()

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }

    @staticmethod
    def validate(engine, mini_batch):
        # 꼭 eval을 불러와야한다.
        engine.model.eval()
        # no_grad 반드시 해줘야한다
        with torch.no_grad():
            x, y = mini_batch.text, mini_batch.label
            x, y = x.to(engine.device), y.to(engine.device)

            x = x[:, :engine.config.max_length]

            y_hat = engine.model(x)

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
        }

    @staticmethod
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
             # loss의 평균
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        training_metric_names = ['loss', 'accuracy', '|param|', '|g_param|']

        # 아직 프린트는 아니고 내부적으로만 들고있음
        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        # If the verbosity is set, progress bar would be shown for mini-batch iterations.
        # Without ignite, you can use tqdm to implement progress bar.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        # If the verbosity is set, statistics would be shown after each epoch.
        # epoch이 끝났을때 출력
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} accuracy={:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                ))

        validation_metric_names = ['loss', 'accuracy']
        
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        # Do same things for validation engine.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss={:.4e} accuracy={:.4f} best_loss={:.4e}'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                    engine.best_loss,
                ))

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss: # If current epoch returns lower validation loss,
            engine.best_loss = loss  # Update lowest validation loss.
            engine.best_model = deepcopy(engine.model.state_dict()) # Update best model weights.

    @staticmethod
    def save_model(engine, train_engine, config, **kwargs):
        torch.save(
            {
                'model': engine.best_model,
                'config': config,
                **kwargs
            }, config.model_fn
        )


class Trainer():

    def __init__(self, config):
        self.config = config

    def train(
        self,
        model, crit, optimizer,
        train_loader, valid_loader,
    ):
        train_engine = MyEngine(
            MyEngine.train,
            model, crit, optimizer, self.config
        )
        validation_engine = MyEngine(
            MyEngine.validate,
            model, crit, optimizer, self.config
        )

        MyEngine.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )
        # train의  1epoch이 끝나면 valid에 1epoch이 시작
        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1) # max_epochs는 벨리데이션 1번씩 n_epochs 만큼 도는것이다

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            run_validation, # function
            validation_engine, valid_loader, # arguments
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            MyEngine.check_best, # function
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        model.load_state_dict(validation_engine.best_model)

        return model
