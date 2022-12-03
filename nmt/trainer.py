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

from nmt.utils import get_grad_norm, get_parameter_norm


# ignite.engine.Engine 상속
# ignite.engine.Engine : event 기반으로, 원하는 process에 대한 callback 함수를 작성하고, event를 등록하고, event가 trigger 되었을때 callback함수를 불러줌
# callback 함수 : 1. 다른 함수의 인자로써 이용되어지는 함수
#                 2. 어떤 이벤트에 의해 호출되어지는 함수
# ignite.engine.Engine은 train하는 engine과 validation하는 engine 2개가 필요하다

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 0
VERBOSE_BATCH_WISE = 0

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
                y_hat.contiguous().view(-1, y_hat.size(-1)), y.contiguous().view(-1)
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
            # gradient clipping 수행
            torch.nn.utils.clip_grad.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_grad_norm,
            )
            
            # gradient descent 수행
            # gpu 사용 시 amp 적용
            if engine.config.gpu_id >= 0:
                engine.scaler.step(engine.optimizer)
                engine.scaler.update()
                
            else:
                engine.optimizer.step()
                
                
        # 단어당 loss값 평균
        loss = float(loss / word_count)
        ppl = np.exp(loss)
        
        return {
            "loss" : loss,
            "ppl" : ppl,
            # amp 사용 시 |param|과 |g_param|이 nan값이나 inf로 발산해버리는 경우 발생
            # 이를 대처하기 위해 an값이나 inf로 발산하면 0으로 대체하여 평균 구할 수 있게끔 함
            "|param|" : p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
            "|g_param|" : p_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.
        }
        
    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()
        
        with torch.no_grad():
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
            
            with autocast():
                y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
                # mini_batch.tgt[0][:, :-1] : <EOS>를 제외한 부분. 모델에 넣어줄 때는 <EOS>를 제거해줘야한다
                # |y_hat| : (batch_size, length, output_size)
            
                loss = engine.crit(
                    y_hat.contiguous().view(-1, y_hat.size(-1)), y.contiguous().view(-1)
                    # |y|, |y_hat| : (batch_size * length)
                )
                
        # 단어의 총 개수 (batch 내부의 sample 별 length의 합)
        word_count = int(mini_batch.tgt[1].sum())
        loss = float(loss / word_count)
        ppl = np.exp(loss)
        
        return {
            "loss" : loss,
            "ppl" : ppl
        }
        
        
    @staticmethod
    def attach(train_engine, validation_engine, training_metric = ["loss", "ppl", "|param|", "|g_param|"], validation_metric = ["loss", "ppl"], verbose = VERBOSE_BATCH_WISE):
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x : x[metric_name]).attach(engine, metric_name)
            
        for metric_name in training_metric:
            attach_running_average(train_engine, metric_name)
            
        if verbose >= VERBOSE_BATCH_WISE:
            progress_bar = ProgressBar(bar_format=None, ncols = 120)
            progress_bar.attach(train_engine, training_metric)
            
        if verbose >= VERBOSE_EPOCH_WISE:
            # train engine의 epoch이 끝날때마다 출력
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics["|param|"]
                avg_g_norm = engine.state.metrics["|g_param|"]
                avg_loss = engine.state.metrics["loss"]
                
                print("Epoch{} - |param| = {:.2e} |g_param| = {:.2e} loss={:.4e} ppl={:.2f}".format(engine.state.epoch,
                                                                                                    avg_p_norm,
                                                                                                    avg_g_norm,
                                                                                                    avg_loss,
                                                                                                    np.exp(avg_loss)
                                                                                                    ))
                
        for metric_name in validation_metric:
            attach_running_average(validation_engine, metric_name)
            
        if verbose >= VERBOSE_BATCH_WISE:
            progress_bar = ProgressBar(bar_format=None, ncols = 120)
            progress_bar.attach(validation_engine, validation_metric)
            
        if verbose >= VERBOSE_EPOCH_WISE:
            # validation epoch이 끝날때마다 출력
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_loss = engine.state.metrics["loss"]
                
                print("Validation - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}".format(avg_loss,
                                                                                                    np.exp(avg_loss),
                                                                                                    engine.best_loss,
                                                                                                    np.exp(engine.best_loss)
                                                                                                    ))
                

    # 중단된 학습 재시작하기
    @staticmethod
    def resume_training(engine, resume_epoch):
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = (resume_epoch -1)


    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            
    
    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        avg_train_loss = train_engine.state.metrics['loss']
        avg_valid_loss = engine.state.metrics['loss']
        
        model_fn = config, model_fn.split(".")
        
        model_fn = model_fn[:-1] + ["%02d" % train_engine.state.epoch,
                                    "%.2f-%.2f" % (avg_train_loss, np.exp(avg_train_loss)),
                                    "%.2f-%.2f" % (avg_valid_loss, np.exp(avg_valid_loss))
                                    ] + [
                                        model_fn[-1]
                                    ] 
                                    
        # 리스트화 된 모델 full name 다시 문자열로 변환
        model_fn = ".".join(model_fn)
        
        torch.save(
            {
                "model" : engine.model.state_dict(),
                "opt" : train_engine.optimizer.state_dict(),
                "config" : config,
                "src_vocab" : src_vocab,
                "tgt_vocab" : tgt_vocab
            }, model_fn
        )
        


class SingleTrainer():
    
    def __init__(self, target_engine_class, config):
        self.target_engine_class = target_engine_class
        self.config = config
        
    def train(
        self,
        model, crit, optimizer,
        train_loader, valid_loader,
        src_vocab, tgt_vocab,
        n_epochs,
        lr_scheduler = None
    ):
        train_engine = self.target_engine_class(
            self.target_engine_class.train,
            model, crit, optimizer,
            lr_scheduler, self.config
        )
        
        valid_engine = self.target_engine_class(
            self.target_engine_class.validate,
            model, crit, optimizer,
            lr_scheduler, self.config
        )
        
        self.target_engine_class.attach(
            train_engine,
            valid_engine,
            verbose = self.config.verbose
        )
        
        def run_validation(engine, validation_engine, valid_loader):
            # validation task이기 때문에 max_epochs를 1로 설정
            # ignite는 따로 validation을 위한 process가 있는것이 아니어서, max_epochs를 1로 설정하여 validation 구현
            validation_engine.run(valid_loader, max_epochs=1)
            
            if engine.lr_scheduler is not None:
                engine.lr_scheduler.step()
                
        # train_engine의 epoch이 끝날때마다(Events.EPOCH_COMPLETED) run_validation 함수 실행    
        train_engine.add_event_handler(Events.EPOCH_COMPLETED, run_validation, valid_engine, valid_loader)
        
        # train_engine이 시작할 때(Events.STARTED) self.target_engine_class의 resume_training 함수 실행
        train_engine.add_event_handler(Events.STARTED, self.target_engine_class.resume_training, self.config.init_epoch)
        
        # valid_engine의 epoch이 끝날때마다(Events.EPOCH_COMPLETED) self.target_engine_class의 check_best 함수 실행
        valid_engine.add_event_handler(Events.EPOCH_COMPLETED, self.target_engine_class.check_best)
        
        # valid_engine의 epoch이 끝날때마다(Events.EPOCH_COMPLETED) self.target_engine_class의 save model 함수 실행
        # 원래는 check_best 함수로 가장 낮은 validation loss(가장 높은 validation ppl)을 가진 모델만 저장할 수 있지만,
        # validation loss가 가장 낮더라도 번역 성능이 가장 낮은 것은 아니기에,
        # 매 epoch마다 model을 저장하고, model fime name에(model_fn) epoch, train loss(train ppl), validation loss(validation ppl)을 함께 저장
        valid_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                       self.target_engine_class.save_model,
                                       train_engine,
                                       self.config,
                                       src_vocab,
                                       tgt_vocab
                                       )
        
        # engine에 run 함수와 함께 인자로 train_loader를 넘겨줌으로써 학습 시작
        train_engine.run(train_loader, max_epochs=n_epochs)
        
        return model
    