from multiprocessing.pool import ApplyResult
from transformers import PreTrainedTokenizer, LineByLineWithRefDataset, LineByLineTextDataset, TextDataset
from torch.utils.data import ConcatDataset, dataset
from .util import prepare_dirs_and_logger
from .config import AnalysisArguments, DataArguments, MiscArgument, SourceMap, TrustMap, get_config, ArticleMap, TwitterMap
import os
from os import path
import warnings
import json
import csv
import re
from multiprocessing import Pool
import random
from typing import Any, List, Optional, Union, Dict, Set, List
from glob import glob
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings('ignore')

def extract_data():
    pass


def get_dataset(
    args: DataArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
) -> Union[LineByLineWithRefDataset, LineByLineTextDataset, TextDataset, ConcatDataset]:
    def _dataset(
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

    
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)
        
    if evaluate:
        return _dataset(args.eval_data_file, args.eval_ref_file)
    elif args.train_data_files:
        return ConcatDataset([_dataset(f) for f in glob(args.train_data_files)])
    else:
        return _dataset(args.train_data_file, args.train_ref_file)


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
                elif args.analysis_data_type!='full' and args.analysis_data_type not in item['data_type']:
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
    args: AnalysisArguments,
    data_args: DataArguments
) -> Dict[str, Dict[str, int]]:
    data_map = ArticleMap () if data_args.data_type == 'article' else TwitterMap()
    row_data = dict()
    for file in data_map.dataset_list:
        analysis_data_file = os.path.join(args.analysis_data_dir, file+'.json')
        with open(analysis_data_file) as fp:
            for line in fp.readlines():
                item = json.loads(line.strip())
                sentence = item['sentence']
                if sentence not in row_data:
                    row_data[sentence] = dict()
                for index, prob in item['word'].items():
                    if int(index) not in row_data[sentence]:
                        row_data[sentence][int(index)] = dict()
                    row_data[sentence][int(index)][file] = prob

    return row_data

def get_mask_model_data(
    args: AnalysisArguments,
    data_args: DataArguments
):
    pass


def main():
    misc_args, model_args, data_args, training_args, adapter_args, analysis_args = get_config()
    prepare_dirs_and_logger(misc_args, model_args,
                            data_args, training_args, adapter_args,analysis_args)
    # extract_data(misc_args, data_args)
    get_analysis_data(analysis_args)


if __name__ == '__main__':
    main()
