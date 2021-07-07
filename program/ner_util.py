import argparse
from scipy.sparse.construct import random
import numpy as np
import json
from typing import Optional, List, Dict, Tuple, Any, NewType
import torch
from transformers import BatchEncoding


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def encode_scores(scores, encodings:BatchEncoding):
    scores = [[score for score in doc] for doc in scores]
    encoded_scores = []
    input_ids = list()
    token_type_ids = list()
    attention_mask = list()
    encoding_list = list()
    error_count = 0

    for (doc_scores, doc_input_ids, doc_attemtion_mask, doc_token_type_ids, doc_offset, doc_encoding) in zip(scores, encodings.input_ids, encodings.attention_mask, encodings.token_type_ids, encodings.offset_mapping, encodings._encodings):
        # create an empty array of -100
        try:
            doc_enc_scores = np.ones(len(doc_offset),dtype=float) * -100
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_scores[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_scores
            encoded_scores.append(doc_enc_scores.tolist())
            input_ids.append(doc_input_ids)
            token_type_ids.append(doc_token_type_ids)
            attention_mask.append(doc_attemtion_mask)
            encoding_list.append(doc_encoding)
        except:
            error_count += 1

    data = {'input_ids':input_ids,'token_type_ids':token_type_ids, 'attention_mask':attention_mask}
    encodings = BatchEncoding(data, encoding_list)


    return encoded_scores, encodings
