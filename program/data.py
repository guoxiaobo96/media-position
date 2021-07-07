from transformers.training_args import TrainingArguments
from torch.utils.data import ConcatDataset, dataset
import os
from os import path
import warnings
import json
from multiprocessing import Pool
import random
from typing import Any, List, Optional, Union, Dict, Set, List
from glob import glob
from sklearn.model_selection import train_test_split

from transformers import (
    PreTrainedTokenizer,
    LineByLineWithRefDataset,
    LineByLineTextDataset,
    TextDataset,
    BertTokenizer
)

from .fine_tune_util import SentenceReplacementDataset, MLMConsistencyDataset
from .util import prepare_dirs_and_logger
from .config import AnalysisArguments, DataArguments, FullArticleMap, MiscArgument, ModelArguments, SourceMap, TrustMap, get_config, ArticleMap, TwitterMap, BaselineArticleMap
from .ner_util import NERDataset, encode_scores


def extract_data():
    pass


def get_dataset(
    training_args: TrainingArguments,
    data_args: DataArguments,
    model_args: ModelArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
) -> Union[LineByLineWithRefDataset, LineByLineTextDataset, TextDataset, ConcatDataset]:
    if training_args.loss_type == 'mlm':
        return mlm_get_dataset(data_args, tokenizer, evaluate, cache_dir)
    elif training_args.loss_type == 'mlm_con':
        return mlm_consistency_get_data(data_args, tokenizer, evaluate)
    elif data_args.data_type in ['sentence_random_replacement', 'sentence_chosen_replacement']:
        return sentence_replacement_get_data(data_args, tokenizer)


def mlm_get_dataset(
    args: DataArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
) -> Union[LineByLineWithRefDataset, LineByLineTextDataset, TextDataset, ConcatDataset]:
    def _mlm_dataset(
        file_path: str,
        ref_path: str = None
    ) -> Union[LineByLineWithRefDataset, LineByLineTextDataset, TextDataset]:
        if args.line_by_line:
            if ref_path is not None:
                if not args.whole_word_mask or not args.mlm:
                    raise ValueError(
                        "You need to set world whole masking and mlm to True for Chinese Whole Word Mask")
                return LineByLineWithRefDataset(
                    tokenizer=tokenizer,
                    file_path=file_path,
                    block_size=args.block_size,
                    ref_path=ref_path,
                )

            return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
        else:
            return TextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=args.block_size,
                overwrite_cache=args.overwrite_cache,
                cache_dir=cache_dir,
            )

    def _reformat_dataset(file_path: str) -> str:
        cache_file_path = file_path+'.cache'
        sentence_list = list()
        with open(file_path, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                item = json.loads(line.strip())
                sentence_list.append(item['original'])
                if 'augmented' in item and item['augmented'] is not None:
                    sentence_list.extend(item['augmented'])
        random.shuffle(sentence_list)
        with open(cache_file_path, mode='w', encoding='utf8') as fp:
            for sentence in sentence_list:
                fp.write(sentence+'\n')
        return cache_file_path

    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.model_max_length)

    if evaluate:
        return _mlm_dataset(args.eval_data_file, args.eval_ref_file)
    elif args.train_data_files:
        return ConcatDataset([_mlm_dataset(f) for f in glob(args.train_data_files)])
    else:
        train_data_file = _reformat_dataset(args.train_data_file)
        return _mlm_dataset(train_data_file, args.train_ref_file)


def mlm_consistency_get_data(
    data_args: DataArguments,
    tokenizer: BertTokenizer,
    evaluate: bool
):
    train_file = os.path.join(data_args.data_path, 'en.train')
    eval_file = os.path.join(data_args.data_path, 'en.valid')


    if not evaluate:
        train_data = {'original_sentence': list(), 'augmented_sentence': list()}
        with open(train_file, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                item = json.loads(line.strip())
                original_sentence = item['original']
                for aug_sentence in item['augmented']:
                    train_data['original_sentence'].append(original_sentence)
                    train_data['augmented_sentence'].append(aug_sentence)
                train_data['original_sentence'].append(original_sentence)
                train_data['augmented_sentence'].append(original_sentence)
        ori_encodings = tokenizer(
            train_data['original_sentence'], padding=False, truncation=True)
        aug_encodings = tokenizer(
            train_data['augmented_sentence'], padding=False, truncation=True)
        dataset = MLMConsistencyDataset(ori_encodings, aug_encodings)
    else:
        if data_args.block_size <= 0:
            data_args.block_size = tokenizer.model_max_length
            # Our input block size will be the max possible for the model
        else:
            data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)
        dataset = LineByLineTextDataset(
            tokenizer=tokenizer, file_path=eval_file, block_size=data_args.block_size)

    return dataset


def sentence_replacement_get_data(
    data_args: DataArguments,
    tokenizer: BertTokenizer
):
    train_file = os.path.join(data_args.data_path, 'en.train')
    eval_file = os.path.join(data_args.data_path, 'en.valid')
    train_data = {'sentence': list(), 'label': list()}
    eval_data = {'sentence': list(), 'label': list()}

    with open(train_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            item = json.loads(line.strip())
            train_data['sentence'].append(item['sentence'])
            train_data['label'].append(item['label'])
    with open(eval_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            item = json.loads(line.strip())
            eval_data['sentence'].append(item['sentence'])
            eval_data['label'].append(item['label'])

    number_label = max(
        len(set(train_data['label'])), len(set(eval_data['label'])))

    train_encodings = tokenizer(
        train_data['sentence'], padding=True, truncation=True)
    val_encodings = tokenizer(
        eval_data['sentence'],  padding=True, truncation=True)

    train_dataset = SentenceReplacementDataset(
        train_encodings, train_data['label'])
    val_dataset = SentenceReplacementDataset(val_encodings, eval_data['label'])

    return train_dataset, val_dataset, number_label


def get_analysis_data(
    args: AnalysisArguments,
) -> Dict[str, Dict[str, int]]:
    data = dict()
    row_data = dict()
    trust_map = TrustMap()
    for file in os.listdir(args.analysis_data_dir):
        analysis_data_file = os.path.join(args.analysis_data_dir, file)
        row_data[file] = dict()
        with open(analysis_data_file) as fp:
            for line in fp.readlines():
                item = json.loads(line.strip())
                if "sentence" in item:
                    continue
                elif args.analysis_data_type != 'full' and args.analysis_data_type not in item['data_type']:
                    continue
                else:
                    row_data[file][item["dataset"]] = item["words"]
    # for file, file_data in row_data.items():
    #     data[file] = dict()
    #     for name in trust_map.republican_datasets_list:
    #         dataset = trust_map.name_to_dataset[name]
    #         if dataset in file_data:
    #             data[file][dataset] = row_data[file][dataset]

    #     dataset = 'vanilla'
    #     data[file][dataset] = row_data[file][dataset]

    #     for name in trust_map.democrat_datasets_list:
    #         dataset = trust_map.name_to_dataset[name]
    #         if dataset in file_data:
    #             data[file][dataset] = row_data[file][dataset]
    # return data

    return row_data


def get_label_data(
    misc_args: MiscArgument,
    analysis_args: AnalysisArguments,
    data_args: DataArguments
) -> Dict[str, Dict[str, int]]:
    data_map = BaselineArticleMap()
    row_data = dict()
    for file in data_map.dataset_list:
        analysis_data_file = os.path.join(
            analysis_args.analysis_data_dir, file+'.json')
        with open(analysis_data_file) as fp:
            count = 0
            for line in fp:
                item = json.loads(line.strip())
                sentence = item['sentence']
                if sentence not in row_data:
                    row_data[sentence] = dict()
                for index, prob in item['word'].items():
                    if int(index) not in row_data[sentence]:
                        row_data[sentence][int(index)] = dict()
                    row_data[sentence][int(index)][file] = prob
                if misc_args.global_debug:
                    count += 1
                    if count == 100:
                        break

    return row_data


def get_mask_score_data(
    analysis_args: AnalysisArguments,
    data_args: DataArguments,
    tokenizer: BertTokenizer
):
    file = analysis_args.analysis_encode_method+'_'+analysis_args.analysis_cluster_method + \
        '_'+analysis_args.analysis_compare_method+'_sentence.json'
    data_path = os.path.join(os.path.join(
        analysis_args.analysis_result_dir, analysis_args.graph_distance), file)
    data_path = '/home/xiaobo/media-position/analysis/article/cluster/dataset/count/term_AgglomerativeClustering_cluster_sentence.json'
    # Tokenize the text into subwords in a label-preserving way
    raw_text = list()
    raw_score = list()
    with open(data_path, mode='r', encoding='utf8') as fp:
        for line in fp:
            item = json.loads(line.strip())
            sentence = item.pop('sentence')
            text_scores = item
            if sentence in ['media_average', 'distance_base', 'cluster_average']:
                continue
            scores = [0 for _ in range(len(text_scores) - 1)]
            texts = [0 for _ in range(len(text_scores) - 1)]
            for position, item in text_scores.items():
                if int(position) >= 0:
                    scores[int(position)] = item[0]
                    texts[int(position)] = item[1]
            raw_score.append(scores)
            raw_text.append(texts)

    train_texts, val_texts, train_scores, val_scores = train_test_split(
        raw_text, raw_score, test_size=.2)
    train_encodings = tokenizer(train_texts, is_split_into_words=True,
                                return_offsets_mapping=True, padding=True, truncation=True)
    val_encodings = tokenizer(val_texts, is_split_into_words=True,
                              return_offsets_mapping=True, padding=True, truncation=True)
    train_scores, train_encodings = encode_scores(
        train_scores, train_encodings)
    val_scores, val_encodings = encode_scores(val_scores, val_encodings)
    train_dataset = NERDataset(train_encodings, train_scores)
    val_dataset = NERDataset(val_encodings, val_scores)

    return train_dataset, val_dataset


def main():
    misc_args, model_args, data_args, training_args, analysis_args = get_config()
    prepare_dirs_and_logger(misc_args, model_args,
                            data_args, training_args, analysis_args)
    # extract_data(misc_args, data_args)
    get_analysis_data(analysis_args)


if __name__ == '__main__':
    main()
