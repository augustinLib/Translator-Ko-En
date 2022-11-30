import numpy as np

import torch
from torch import optim
import torch.nn as nn
# for mixed Precision training
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from utils import get_parameter_norm, get_grad_norm


# ignite.engine.Engine 상속
# ignite.engine.Engine : event 기반으로, 원하는 process에 대한 callback 함수를 작성하고, event를 등록하고, event가 trigger 되었을때 callback함수를 불러줌
# callback 함수 : 1. 다른 함수의 인자로써 이용되어지는 함수
#                 2. 어떤 이벤트에 의해 호출되어지는 함수
# ignite.engine.Engine은 train하는 engine과 validation하는 engine 2개가 필요하다


class MaximumLikelihoodEstimationEngine(Engine):
    def __init__(self, func, model, crit, optimizer, lr_scheduler, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        
        super().__init__(func)
        
        self.best_loss = np.inf
        self.scaler = GradScaler()
        
    @staticmethod
    def train(engine, mini_batch):
        engine.model.train()
        # gradient accumulation을 위해 매 iteration마다 gradient update를 하는것이 아닌,
        # engine.config.iteration_per_update에 지정된 횟수의 iteration이 지나면 gradient update
        # engine.config.iteration_per_update이 1이면 매 iteration마다 update(gradient accumulation을 사용하지 않음)
        if engine.state.iteration % engine.config.iteration_per_update == 1 or \
            engine.config.iteration_per_update == 1:
                engine.optimizer.zero_grad()
        
        # 현재 model의 첫번째 weight parameter의 device
        device = next(engine.model.parameters()).device
        mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
        mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])
        # mini_batch.src[0].to(device) : 실제 tensor가 들어가있음
        # mini_batch.src[1]            : length가 들어가있음
        # mini_batch.tgt[0].to(device) : 실제 tensor가 들어가있음
        # mini_batch.tgt[1]            : length가 들어가있음
        
        x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]
        # |x| : (batch_size, length)
        # |y| : (batch_size, length)
        # mini_batch.tgt[0][:, 1:] : <BOS>를 제외한 부분, 추후 <EOS>를 제외하고 model에 넣은 y_hat과 비교하여 loss 비교
        
        
        # Mixed Precision Training
        with autocast():
            y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
            # mini_batch.tgt[0][:, :-1] : <EOS>를 제외한 부분. 모델에 넣어줄 때는 <EOS>를 제거해줘야한다
            # |y_hat| : (batch_size, length, output_size)
            
            loss = engine.crit(
                y_hat.contiguous.view(-1, y_hat.size(-1)), y.contiguous().view(-1)
                # |y|, |y_hat| : (batch_size * length)
            )
            # accumulated된 loss를 batch size로 나눠주고, gradient accumulation을 위해 설정한 지정된 횟수의 iteration 수로 나눠줌
            backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)
          
        # FP16으로 계산되어 제대로 표현되지 않는 값들을 방지하고자 loss scaling 적용
        # 단 gpu에서만 동작함
        if engine.config.gpu_id >= 0:
            engine.scaler.scale(backward_target).backward()
        else:
            backward_target.backward()
        
        # 단어의 총 개수 (batch 내부의 sample 별 length의 합)
        word_count = int(mini_batch.tgt[1].sum())
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))
        
        # gradient accumulation을 위해 매 iteration마다 gradient update를 하는것이 아닌,
        # engine.config.iteration_per_update에 지정된 횟수의 iteration이 지나면 gradient update
        if engine.state.iteration % engine.config.iteration_per_update == 0:
            # gradient clipping
            torch.nn.utils.clip_grad.clip_grad_norm(
                engine.model.parameters(),
                engine.config.max_grad_norm,
            )