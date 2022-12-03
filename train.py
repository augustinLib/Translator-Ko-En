import argparse
import pprint

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, BatchSampler


from nmt.loader import SPECIAL_TOKENS
from nmt.loader import read_txt, get_vocab, TranslationDataset, TranslatorCollator, SequenceLengthBasedBatchSampler
from nmt.models.seq2seq import Seq2Seq
from nmt.trainer import SingleTrainer, MaximumLikelihoodEstimationEngine


def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',
            required=True,
            help='Model file name to continue.'
        )

    p.add_argument(
        '--model_fn',
        required=not is_continue,
        help='Model file name to save. Additional information would be annotated to the file name.'
    )
    p.add_argument(
        '--train',
        required=not is_continue,
        help='Training set file name except the extention. (ex: train.en --> train)'
    )
    p.add_argument(
        '--valid',
        required=not is_continue,
        help='Validation set file name except the extention. (ex: valid.en --> valid)'
    )
    p.add_argument(
        '--lang',
        required=not is_continue,
        help='Set of extention represents language pair. (ex: en + ko --> enko)'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s'
    )
    p.add_argument(
        '--off_autocast',
        action='store_true',
        help='Turn-off Automatic Mixed Precision (AMP), which speed-up training.',
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=20,
        help='Number of epochs to train. Default=%(default)s'
    )
    p.add_argument(
        '--verbose',
        type=int,
        default=2,
        help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s'
    )
    p.add_argument(
        '--init_epoch',
        required=is_continue,
        type=int,
        default=1,
        help='Set initial epoch number, which can be useful in continue training. Default=%(default)s'
    )

    p.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Maximum length of the training sequence. Default=%(default)s'
    )
    p.add_argument(
        '--dropout',
        type=float,
        default=.2,
        help='Dropout rate. Default=%(default)s'
    )
    p.add_argument(
        '--word_vec_size',
        type=int,
        default=512,
        help='Word embedding vector dimension. Default=%(default)s'
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='Hidden size of LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--n_layers',
        type=int,
        default=4,
        help='Number of layers in LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=5.,
        help='Threshold for gradient clipping. Default=%(default)s'
    )
    p.add_argument(
        '--iteration_per_update',
        type=int,
        default=1,
        help='Number of feed-forward iterations for one parameter update. Default=%(default)s'
    )

    p.add_argument(
        '--lr',
        type=float,
        default=1.,
        help='Initial learning rate. Default=%(default)s',
    )

    p.add_argument(
        '--lr_step',
        type=int,
        default=1,
        help='Number of epochs for each learning rate decay. Default=%(default)s',
    )
    p.add_argument(
        '--lr_gamma',
        type=float,
        default=.5,
        help='Learning rate decay rate. Default=%(default)s',
    )
    p.add_argument(
        '--lr_decay_start',
        type=int,
        default=10,
        help='Learning rate decay start at. Default=%(default)s',
    )

    p.add_argument(
        '--use_adam',
        action='store_true',
        help='Use Adam as optimizer instead of SGD. Other lr arguments should be changed.',
    )
    p.add_argument(
        '--use_radam',
        action='store_true',
        help='Use rectified Adam as optimizer. Other lr arguments should be changed.',
    )

    p.add_argument(
        '--rl_lr',
        type=float,
        default=.01,
        help='Learning rate for reinforcement learning. Default=%(default)s'
    )
    p.add_argument(
        '--rl_n_samples',
        type=int,
        default=1,
        help='Number of samples to get baseline. Default=%(default)s'
    )
    p.add_argument(
        '--rl_n_epochs',
        type=int,
        default=10,
        help='Number of epochs for reinforcement learning. Default=%(default)s'
    )
    p.add_argument(
        '--rl_n_gram',
        type=int,
        default=6,
        help='Maximum number of tokens to calculate BLEU for reinforcement learning. Default=%(default)s'
    )
    p.add_argument(
        '--rl_reward',
        type=str,
        default='gleu',
        help='Method name to use as reward function for RL training. Default=%(default)s'
    )

    p.add_argument(
        '--use_transformer',
        action='store_true',
        help='Set model architecture as Transformer.',
    )
    p.add_argument(
        '--n_splits',
        type=int,
        default=8,
        help='Number of heads in multi-head attention in Transformer. Default=%(default)s',
    )

    config = p.parse_args()

    return config


def get_loaders(config:argparse.ArgumentParser, is_dsl:bool = False):
    """
    get data from data directory
    return train dataloader and validation dataloader

    Args:
        config (argparse.ArgumentParser): _description_
        is_dsl (bool, optional): _description_. Defaults to False.

    Returns:
        tuple: _description_
    """
    
    # read_txt() -> list of tuple(splitted l1 list, splitted l2 list)
    train_texts = read_txt(config.train, (config.lang[:2], config.lang[-2:]), max_length=config.max_length)
    valid_texts = read_txt(config.valid, (config.lang[:2], config.lang[-2:]), max_length=config.max_length)
    
    
    src_vocab = get_vocab([src for src, _ in train_texts])
    tgt_vocab = get_vocab([tgt for _, tgt in train_texts])
    
    
    train_dataset = TranslationDataset(
            train_texts, src_vocab, tgt_vocab,
            special_token_at_both=is_dsl,
        )
    
    valid_dataset = TranslationDataset(
            valid_texts, src_vocab, tgt_vocab,
            special_token_at_both=is_dsl,
        )
    
    
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=SequenceLengthBasedBatchSampler(
            train_texts,
            config.batch_size
        ),
        collate_fn=TranslatorCollator(),
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=TranslatorCollator(),
    )
    
    return train_dataloader, valid_dataloader, src_vocab, tgt_vocab


# get model
def get_model(input_size, output_size, config):
    model = Seq2Seq(input_size=input_size,
                    word_vec_size=config.word_vec_size,
                    hidden_size=config.hidden_size,
                    output_size=output_size,
                    n_layers=config.n_layers,
                    dropout_p=config.dropout)
    
    return model



def get_crit(output_size, pad_index):
    """

    Args:
        output_size (_type_): size of word vector that model have to predict
        pad_index (_type_): _description_

    Returns:
        _type_: _description_
    """
    # loss weight : 예측을 할 때 target y에 pad가 들어가있을거임(autoregressive)
    # pad가 있을 때, pad를 맞추는지 맞추지 못하는지에 대해서는 평가하지 않아야 함
    # 이를 위해 loss를 계산할 때 weight를 줘서 pad의 위치에는 weight를 0을 줘서(default는 모두 1)
    # pad 부분에 대해서는 loss를 계산하지 않게끔 함
    loss_weight = torch.ones(output_size)
    loss_weight[pad_index] = 0.
    
    # loss function 선언 시 weight parameter에 위에서 선언한 loss_weight 전달
    # reduction=sum : loss 값을 모두 더하여 반환
    # tranier에서 gradient accumulation을 위한 작업인
    # accumulated된 loss를 batch size로 나눠주고, gradient accumulation을 위해 설정한 지정된 횟수의 iteration 수로 나눠주는 작업을 수행했기에
    # loss function 자체에서는 loss의 합을 나눠줘서 반환하고, 이를 trainer.py에서 나눠서 처리하게끔 함
    crit = nn.NLLLoss(weight=loss_weight,
                      reduction="sum")
    
    return crit
        


def get_optimizer(model, config):
    if config.use_adam:
        if config.use_transformer:
            optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .98))
        else: # case of rnn based seq2seq.
            optimizer = optim.Adam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

    return optimizer


def get_scheduler(optimizer, config):
    if config.lr_step > 0:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[i for i in range(
                max(0, config.lr_decay_start - 1),
                (config.init_epoch - 1) + config.n_epochs,
                config.lr_step
            )],
            gamma=config.lr_gamma,
            last_epoch=config.init_epoch - 1 if config.init_epoch > 1 else -1,
        )
    else:
        lr_scheduler = None

    return lr_scheduler
    

def main(config, model_weight = None, opt_weight = None):
    # pprint : 예쁘게 인쇄해주는 역할
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    
    print_config(config)
    
    
    train_loader, valid_loader, src_vocab, tgt_vocab = get_loaders(config=config,
                                                                   is_dsl=False)
    
    # input, output size = source data의 vocab의 길이와 target data의 vocab의 길이
    input_size, output_size = len(src_vocab), len(tgt_vocab)
    model = get_model(input_size=input_size,
                      output_size=output_size,
                      config=config)
    
    crit = get_crit(output_size=output_size,
                    pad_index=SPECIAL_TOKENS.PAD_idx)
    
    # continue learning
    if model_weight is not None:
        model.load_state_dict(model_weight)
        
    
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)
        
    optimizer = get_optimizer(model=model,
                              config=config)
    
    
    if opt_weight is not None and (config.use_adam or config.use_radam):
        optimizer.load_state_dict(opt_weight)

    lr_scheduler = get_scheduler(optimizer, config)
    

    if config.verbose >= 2:
        print(model)
        print(crit)
        print(optimizer)
        
    trainer = SingleTrainer(MaximumLikelihoodEstimationEngine,
                            config=config)
    trainer.train(model=model,
                  crit=crit,
                  optimizer=optimizer,
                  train_loader=train_loader,
                  valid_loader=valid_loader,
                  src_vocab=src_vocab,
                  tgt_vocab=tgt_vocab,
                  n_epochs=config.n_epochs,
                  lr_scheduler=lr_scheduler)
    

    

if __name__ == '__main__':
    config = define_argparser()
    main(config)