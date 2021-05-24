import argparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
import numpy as np
import json
from typing import Optional, List, Dict, Tuple, Any, NewType


class NerBertDataset:
    def __init__(self, data_path:str, tokenizer:BertTokenizer, max_len):

        """
        Takes care of the tokenization and ID-conversion steps
        for prepping data for BERT.
        """

        pad_tok = tokenizer.vocab["[PAD]"]
        sep_tok = tokenizer.vocab["[SEP]"]
        tokenized_texts = list()
        # Tokenize the text into subwords in a label-preserving way
        with open(data_path, mode='r',encoding='utf8') as fp:
            for line in fp:
                item  = json.loads(line.strip())
                sentence = item.pop('sentence')
                text_scores = item
                if sentence in ['media_average','distance_base','cluster_average']:
                    continue
                tokenized_texts.append(tokenize_and_preserve_labels(sentence,text_scores, tokenizer))

        self.sentences = [text[0] for text in tokenized_texts]
        self.tokens = [["[CLS]"] + text[1] for text in tokenized_texts]
        self.scores = [[1] + text[2] for text in tokenized_texts]

        # Convert tokens to IDs
        self.input_ids = pad_sequences(
            [tokenizer.convert_tokens_to_ids(txt) for txt in self.tokens],
            maxlen=max_len,
            dtype="long",
            truncating="post",
            padding="post",
        )

        # Convert tags to IDs
        self.scores = pad_sequences(
            [score for score in self.scores],
            maxlen=max_len,
            value=1,
            padding="post",
            dtype="float",
            truncating="post",
        )

        # Swaps out the final token-label pair for ([SEP], O)
        # for any sequences that reach the MAX_LEN
        for voc_ids, scores_ids in zip(self.input_ids, self.scores):
            if voc_ids[-1] == pad_tok:
                continue
            else:
                voc_ids[-1] = sep_tok
                scores_ids[-1] = 1

        # Place a mask (zero) over the padding tokens
        self.attn_masks = [[float(i > 0) for i in ii] for ii in self.input_ids]


def tokenize_and_preserve_labels(sentence:str, text_scores:Dict, tokenizer:BertTokenizer):

    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    scores = []
    
    ordered_sentence = [[] for _ in range(len(text_scores)-1)]
    for position, info in text_scores.items():
        if position == '-1':
            continue
        ordered_sentence[int(position)] = info
    for info in ordered_sentence:
        score, word = info
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        scores.extend([score] * n_subwords)

    return sentence, tokenized_sentence, scores


def main():

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    ner_bert_dataset = NerBertDataset('/home/xiaobo/media-position/analysis/article/cluster/dataset/count/term_AgglomerativeClustering_cluster_sentence.json', tokenizer, 512)
    print('test')

if __name__ == '__main__':
    main()