import random
from argparse import Namespace

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import torchtext
from torchtext.vocab import build_vocab_from_iterator

# 특수 기호(symbol)와 인덱스를 정의
SPECIAL_TOKENS = {
    'PAD': '<PAD>',
    'UNK': '<UNK>',
    'BOS': '<BOS>',
    'EOS': '<EOS>',
    'PAD_idx': 0,
    'UNK_idx': 1,
    'BOS_idx': 2,
    'EOS_idx': 3,
}
SPECIAL_TOKENS = Namespace(**SPECIAL_TOKENS)

# txt file로 되어있는 한국어, 영어 파일 불러오기
def read_txt(fn:str, exts:str, max_length= 256):
    """
    read txt file for train   

    Args:
        fn (str): file name
        exts (str): language type (Ex. ko, en ....)
        max_length (int, optional): maximum token counts in setence

    Returns:
        list of tuple(splitted l1 list, splitted l2 list)
    """
    
    null_count = 0
    max_count = 0
    available_count = 0
    
    parallel_lines = []
    
    # tsv 파일 불러오기
    with open(fn + "." + exts[0], "r") as f1:
        with open(fn + "." + exts[1], "r") as f2:
            # read() - 파일 내용 전체를 가져와 문자열로 반환. readlines()와 마찬가지로 파일 내용 전체를 읽고, 파일 내용 전체를 하나의 문자열로 반환 
            # 각각의 줄은 '\n' 문자로 구분됨
            # readline() - 파일의 한 줄을 가져와 문자열로 반환. 파일 포인터는 그 다음줄로 이동
            # readlines() - 파일 내용 전체를 가져와 리스트로 반환. 각 줄은 문자열 형태로 리스트의 요소로 저장됨
            for l1, l2 in zip(f1.readlines(), f2.readlines()):
                # str.strip() : 문자열 양 옆 공백 제거
                # l1이나 l2 중 양 옆 공백을 제거한 문자열이 공백일 경우
                # null_count 추가 후 무시
                if l1.strip() == "" or l2.strip() == "":
                    null_count += 1
                    continue
                
                # str.split() : 공백 기준으로 문자열 분절 후 list 형태로 반환
                # l1이나 l2 중 문장 token 개수가 max_length보다 클 경우
                # max_count 추가 후 무시
                if len(l1.split()) > max_length or len(l2.split()) > max_length:
                    max_count += 1
                    continue
                
                # 문제가 없으면 l1, l2 각각 양 옆 공백 제거 후, split()으로 문자열로 반환
                # 이후 available_count 추가
                # parallel_lines = [(splitted l1 list, splitted l2 list)]
                parallel_lines += [(
                    l1.strip().split(),
                    l2.strip().split()
                )]
                available_count += 1
    
                
    print(f"Read {fn}.{exts[0]} and {fn}.{exts[1]}")
    print(f"Null data count : {null_count}")
    print(f"Over max_length : {max_length}")
    print(f"Available data count : {available_count}")
    
    return parallel_lines

def get_vocab(texts:list, min_freq = 1):
    """
    get Vocab from text list

    Args:
        texts (list): list of tuple(splitted l1 list, splitted l2 list)
        min_freq (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: torchtext.vocab.vocab.Vocab
    """
    # torchtext.vocab.build_vocab_from_iterator
    # 토큰의 목록 또는 반복자(iterator)를 받아 torchtext.vocab.vocab.Vocab return
    # parameters
    # iterator – Iterator used to build Vocab. Must yield list or iterator of tokens
    # min_freq – The minimum frequency needed to include a token in the vocabulary.
    # specials – Special symbols to add. The order of supplied tokens will be preserved.
    # special_first – Indicates whether to insert symbols at the beginning or at the end.
    vocab = build_vocab_from_iterator(texts,
                                      min_freq=min_freq,
                                      specials=[SPECIAL_TOKENS.PAD,
                                                SPECIAL_TOKENS.UNK,
                                                SPECIAL_TOKENS.BOS,
                                                SPECIAL_TOKENS.EOS],
                                      special_first=True)
    
    # Vocab.set_default_index : parameter로 받은 index는 SPECIAL_TOKENS.UNK_idx인 1 이고,
    # OOV token이 나올 경우 해당 index 반환됨
    # 만약 set_default_index() 처리를 해주지 않으면 RuntimeError 발생됨
    # Value of default index. This index will be returned when OOV token is queried.
    vocab.set_default_index(SPECIAL_TOKENS.UNK_idx)
    return vocab

class TranslationDataset(Dataset):
    """
    Dataset class for machine translator
    inherited class from torch.utils.data.Dataset
    """
    def __init__(self, texts:list, src_vocab:torchtext.vocab.Vocab, tgt_vocab:torchtext.vocab.Vocab, special_token_at_both:bool = False): 
        """_summary_
        Dataset class for machine translator
        inherited class from torch.utils.data.Dataset
        Args:
            texts (list): data that needs to be converted to a Dataset
            src_vocab (torchtext.vocab.Vocab): source data vocab
            tgt_vocab (torchtext.vocab.Vocab): target data vocab
            special_tokens_at_both (bool, optional): _description_. Defaults to False.
        """
        self.texts = texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.special_token_at_both = special_token_at_both
        
        for i in range(len(texts)):
            if self.special_token_at_both:
                src_text = [SPECIAL_TOKENS.BOS] + texts[i][0] + [SPECIAL_TOKENS.EOS]
                
            else:
                src_text = texts[i][0]
            
            tgt_text = [SPECIAL_TOKENS.BOS] + texts[i][1] + [SPECIAL_TOKENS.EOS]
            
            texts[i] = (torch.tensor(src_vocab(src_text), dtype=torch.long),
                        torch.tensor(tgt_vocab(tgt_text), dtype=torch.long))
    
    
    def __len__(self) :
        """
        return length of text
        
        Returns:
            int: length of text
        """
        return len(self.texts)
    
    
    def __getitem__(self, index):
        return {"src" : self.texts[index][0],
                "tgt" : self.texts[index][1]}
        

# Dataloader의 collate_fn의 parameter값으로 return값을 넘겨준다.
# collate_fn (Callable, optional) – merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.

class TranslatorCollator():
    # TranslatorCollator callable로 변환
    def __call__(self, samples):
        """
        convert TranslatorCollator to callable object
        Args:
            samples (_type_): _description_

        Returns:
            Namespace: _description_
        """
        src = sorted([(sample["src"], sample["src"].size(-1)) for sample in samples],
                     key = lambda x : x[1],
                     reverse=True
                     )
        
        tgt = sorted([(sample["tgt"], sample["tgt"].size(-1)) for sample in samples],
                     key = lambda x : x[1],
                     reverse=True)
        
        return_value = {'src': (pad_sequence([s[0] for s in src], batch_first=True, padding_value=SPECIAL_TOKENS.PAD_idx), 
                                torch.tensor([s[1] for s in src], dtype=torch.long)),
                        'tgt': (pad_sequence([t[0] for t in tgt], batch_first=True, padding_value=SPECIAL_TOKENS.PAD_idx,),
                                torch.tensor([t[1] for t in tgt], dtype=torch.long))
        }

        return Namespace(**return_value)
    
# Dataset은 idx로 데이터를 가져오도록 설계 되었다. 이 때 Sampler는 이 idx 값을 컨트롤하는 방법.
# sequence의 길이가 모두 다르기 때문에 문장 길이 기준으로 정렬한뒤,
# sampler를 사용할 때 Dataloader의 shuffle 파라미터는 False를 해야 함
class SequenceLengthBasedBatchSampler():
    """_summary_
    control batch sampling method
    sampling batch by sequence length
    """
    def __init__(self, texts, batch_size):
        self.batch_size = batch_size
        self.lens = [len(text[1]) for text in texts]
        self.index = [i for i in range(len(texts))]

        # 문장 길이 별 정렬
        temp = sorted(zip(self.lens, self.index), key=lambda x: x[0])
        self.sorted_lens = [x[0] for x in temp]
        self.sorted_index  = [x[1] for x in temp]

    def __iter__(self):
        batch_indice = [i for i in range(0, len(self.lens), self.batch_size)]
        random.shuffle(batch_indice)

        for i, batch_idx in enumerate(batch_indice):
            ret = self.sorted_index[batch_idx:batch_idx + self.batch_size]

            yield ret
        
    def __len__(self):
        return len(self.lens) // self.batch_size