from transformers.training_args import TrainingArguments
from torch.utils.data import ConcatDataset
import os
import json
from multiprocessing import Pool
import random
from typing import Optional, Union, Dict
from glob import glob

from transformers import (
    PreTrainedTokenizer,
    LineByLineWithRefDataset,
    LineByLineTextDataset,
    TextDataset,
    BertTokenizer
)

from .fine_tune_util import ClassConsistencyDataset,  MLMConsistencyDataset
from .config import AnalysisArguments, DataArguments, MiscArgument, ModelArguments, BaselineArticleMap
from .masked_token_util import ClassifyDataset


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
    basic_loss_type = training_args.loss_type.split('_')[0]
    add_loss_type = None
    if len(training_args.loss_type.split('_')) > 1:
        add_loss_type = training_args.loss_type.split('_')[1]
    if basic_loss_type == 'mlm':
        if add_loss_type is None:
            return mlm_get_dataset(data_args, tokenizer, evaluate, cache_dir)
        else:
            return mlm_consistency_get_data(data_args, tokenizer, evaluate)
    elif basic_loss_type == 'class':
        if add_loss_type is None:
            return class_get_data(data_args, tokenizer, evaluate)
        else:
            return class_consistency_get_data(data_args, tokenizer, evaluate)


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
        train_data = {'original_sentence': list(),
                      'augmented_sentence': list()}
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
            data_args.block_size = min(
                data_args.block_size, tokenizer.model_max_length)
        dataset = LineByLineTextDataset(
            tokenizer=tokenizer, file_path=eval_file, block_size=data_args.block_size)

    return dataset


def class_get_data(
    data_args: DataArguments,
    tokenizer: BertTokenizer,
    evaluate: bool
):
    train_file = os.path.join(data_args.data_path, 'en.train')
    eval_file = os.path.join(data_args.data_path, 'en.valid')

    if not evaluate:
        file = train_file
    else:
        file = eval_file

    data = {'sentence': [], 'label': []}
    with open(file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            item = json.loads(line.strip())
            sentence = item['sentence']
            label = item['label']
            data['sentence'].append(sentence)
            data['label'].append(label)

    encodings = tokenizer(
        data['sentence'], padding=False, truncation=True)
    dataset = ClassifyDataset(encodings, data['label'])

    return dataset


def class_consistency_get_data(
    data_args: DataArguments,
    tokenizer: BertTokenizer,
    evaluate: bool
):
    train_file = os.path.join(data_args.data_path, 'en.train')
    eval_file = os.path.join(data_args.data_path, 'en.valid')

    if not evaluate:
        train_data = {'original_sentence': list(), 'augmented_sentence': list(
        ), 'original_label': list(), 'augmented_label': list()}
        with open(train_file, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                item = json.loads(line.strip())
                original_sentence = item['original']['sentence']
                original_label = item['original']['label']
                for aug_item in item['augmented']:
                    train_data['original_sentence'].append(original_sentence)
                    train_data['original_label'].append(original_label)
                    train_data['augmented_sentence'].append(
                        aug_item['sentence'])
                    train_data['augmented_label'].append(aug_item['label'])

                train_data['original_sentence'].append(original_sentence)
                train_data['original_label'].append(original_label)
                train_data['augmented_sentence'].append(original_sentence)
                train_data['augmented_label'].append(original_label)

        ori_encodings = tokenizer(
            train_data['original_sentence'], padding=False, truncation=True)
        aug_encodings = tokenizer(
            train_data['augmented_sentence'], padding=False, truncation=True)
        dataset = ClassConsistencyDataset(
            ori_encodings, aug_encodings, train_data['original_label'], train_data['augmented_label'])
    else:
        if data_args.block_size <= 0:
            data_args.block_size = tokenizer.model_max_length
            # Our input block size will be the max possible for the model
        else:
            data_args.block_size = min(
                data_args.block_size, tokenizer.model_max_length)
        dataset = LineByLineTextDataset(
            tokenizer=tokenizer, file_path=eval_file, block_size=data_args.block_size)

    return dataset


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


def main():
    pass


if __name__ == '__main__':
    main()
