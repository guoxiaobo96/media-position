from ast import Str
from .config import DataArguments, DataAugArguments, FullArticleMap, MiscArgument, ModelArguments, PredictArguments, TrainingArguments, AnalysisArguments, BaselineArticleMap
from .model import MLMModel,  ClassifyModel
from .data import get_dataset, get_label_data
import numpy as np
from transformers import PreTrainedTokenizerBase
import transformers
import torch


class TokenChecker(object):
    def __init__(self, model_type:str, tokenizer:PreTrainedTokenizerBase ) -> None:
        self._model_type = model_type
        self._tokenizer = tokenizer
        self._check_func = None
        if self._model_type in ['bert-base-uncased','bert-base-cased']:
            self._check_func = self._bert_check
        elif self._model_type == 'roberta-base':
            self._check_func = self._roberta_check

    def check_token(self, token:str) -> bool:
        return self._check_func(token)
    
    def _bert_check(self, token:str) -> bool:
        if token[0].isalpha():
            return True
        return False

    def _roberta_check(self, token:str) -> bool:
        if token[0] == " ":
            return True
        return False
        