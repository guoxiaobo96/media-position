import argparse
from scipy.sparse.construct import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
import numpy as np
import json
from typing import Optional, List, Dict, Tuple, Any, NewType
import torch
from transformers.tokenization_bert_fast import BertTokenizerFast
from transformers import BatchEncoding


class SentenceReplacementDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
