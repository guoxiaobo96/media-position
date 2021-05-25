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


def main():
    from sklearn.model_selection import train_test_split
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', do_lower_case=False)
    data_path = '/home/xiaobo/media-position/analysis/article/cluster/dataset/count/term_AgglomerativeClustering_cluster_sentence.json'
    tokenized_texts = list()
    # Tokenize the text into subwords in a label-preserving way
    raw_text = list()
    raw_score = list()
    with open(data_path, mode='r',encoding='utf8') as fp:
        for line in fp:
            item  = json.loads(line.strip())
            sentence = item.pop('sentence')
            text_scores = item
            if sentence in ['media_average','distance_base','cluster_average']:
                continue

            scores = [0 for _ in range(len(text_scores) - 1)]
            texts = [0 for _ in range(len(text_scores) - 1)]
            for position, item in text_scores.items():
                if int(position) >= 0:
                    scores[int(position)] = item[0]
                    texts[int(position)] = item[1]
            raw_score.append(scores)
            raw_text.append(texts)
    
    train_texts, val_texts, train_scores, val_scores = train_test_split(raw_text, raw_score, test_size=.2, random_state=0)
    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    train_scores = encode_scores(train_scores, train_encodings, tokenizer)
    val_scores = encode_scores(val_scores, val_encodings, tokenizer)
    train_dataset = NERDataset(train_encodings, train_scores)
    val_dataset = NERDataset(val_encodings, val_scores)




if __name__ == '__main__':
    main()