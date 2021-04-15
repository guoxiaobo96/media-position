from multiprocessing.pool import ApplyResult
from transformers import PreTrainedTokenizer, LineByLineWithRefDataset, LineByLineTextDataset, TextDataset
from torch.utils.data import ConcatDataset, dataset
from .util import prepare_dirs_and_logger
from .config import AnalysisArguments, DataArguments, MiscArgument, SourceMap, TrustMap, get_config
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

# def extract_data(
#     misc_args: MiscArgument,
#     data_args: DataArguments
# ) -> None:
#     dataset_map = SourceMap
#     debug = misc_args.global_debug
#     original_dir = data_args.raw_data_dir
#     original_dir = (
#         os.path.join(data_args.raw_data_dir,'processed') if data_args.data_type=="chosen" else os.path.join(data_args.raw_data_dir,'original')
#     )
#     target_dir = data_args.data_dir
#     dataset = data_args.dataset


#     processed_data: Dict[str,Dict] = dict()
#     result_dict: Dict[str, List] = dict()
    

#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)

#     dataset_list = (
#         dataset_map.country_to_name[dataset] if dataset in dataset_map.country_to_name else [
#             dataset_map.dataset_to_name[dataset]]
#     )

#     dataset_list = [dataset_map.name_to_dataset[name] for name in dataset_list]
#     if data_args.data_type == 'chosen':
#         if debug:
#             for _, subdataset in enumerate(dataset_list[:2]):
#                 result_dict[subdataset] = []
#                 original_data_dir = os.path.join(original_dir, subdataset)
#                 for _, file_name in enumerate(os.listdir(original_data_dir)[:6]):
#                     if ('back' in file_name and data_args.dataset!='background') or ('back' not in file_name and data_args.dataset=='background'):
#                         continue
#                     file = os.path.join(original_data_dir, file_name)
#                     result_dict[subdataset].append(_extract_chosen_data(file,data_args.dataset,))
#         else:
#             with Pool(processes=7) as pool:
#                 for subdataset in dataset_list:
#                     result_dict[subdataset] = []
#                     original_data_dir = os.path.join(original_dir, subdataset)
#                     for file_name in os.listdir(original_data_dir):
#                         if ('back' in file_name and data_args.dataset!='background') or ('back' not in file_name and data_args.dataset=='background'):
#                             continue
#                         file = os.path.join(original_data_dir, file_name)
#                         result = pool.apply_async(func=_extract_chosen_data, args=(file,data_args.dataset,))
#                         # result_dict[subdataset].append(pool.apply_async(func=_extract_data, args=(file,)))
#                         result_dict[subdataset].append(result)

#                 pool.close()
#                 pool.join()
#     elif data_args.data_type == 'full':
#         if debug:
#             for _, subdataset in enumerate(dataset_list):
#                 result_dict[subdataset] = []
#                 original_data_file = os.path.join(original_dir, subdataset)+'_tweets_csv_hashed.csv'
#                 result_dict[subdataset].append(_extract_full_data(original_data_file,misc_args))
#         else:
#             with Pool(processes=7) as pool:
#                 for subdataset in dataset_list:
#                     result_dict[subdataset] = []
#                     original_data_file = os.path.join(original_dir, subdataset)+'_tweets_csv_hashed.csv'
#                     result = pool.apply_async(func=_extract_full_data, args=(original_data_file,misc_args))
#                     result_dict[subdataset].append(result)

#                 pool.close()
#                 pool.join()

#     for subdataset, result_list in result_dict.items():
#         processed_data[subdataset] = dict()
#         for result_temp in result_list:
#             result = (
#                 result_temp if debug else result_temp.get()
#             )
#             for lang, texts in result.items():
#                 if lang not in processed_data[subdataset]:
#                     processed_data[subdataset][lang] = set()
#                 processed_data[subdataset][lang] = processed_data[subdataset][lang] | texts
#     if len(processed_data) > 1:
#         processed_data[dataset] = dict()
#         for subdataset, subdataset_result_list in processed_data.items():
#             if subdataset == dataset:
#                 continue
#             for lang, texts in subdataset_result_list.items():
#                 if lang not in processed_data[dataset]:
#                     processed_data[dataset][lang] = set()
#                 processed_data[dataset][lang] = processed_data[dataset][lang] | texts
#     for subdataset, subdataset_texts in processed_data.items():
#         if data_args.dataset == 'background' and subdataset != 'background':
#             continue
#         subdataset_target_dir = os.path.join(os.path.join(target_dir, subdataset), data_args.data_type)
#         if not os.path.exists(subdataset_target_dir):
#             os.makedirs(subdataset_target_dir)

#         most_language_count = 0
#         main_language = str()
#         language_list = ['en']
#         for language, texts in subdataset_texts.items():
#             if len(texts) < 100 or language == 'en':
#                 continue
#             else:
#                 if most_language_count < len(texts):
#                     most_language_count = len(texts)
#                     main_language = language
#             language_list = (
#                 (['en', main_language] if main_language != '' else ['en']
#                 ) if dataset!= 'background' else list(subdataset_texts.keys())
#             )
#         for lang in language_list:
#             texts = processed_data[subdataset][lang]
#             random.seed(123)
#             texts = list(texts)
#             random.shuffle(texts)
#             train_number = int(len(texts)*0.7)
#             train_file = os.path.join(subdataset_target_dir, lang+'.train')
#             with open(train_file, mode='w', encoding='utf8') as fp:
#                 for text in texts[:train_number]:
#                     fp.write(text+'\n')
#             eval_file = os.path.join(subdataset_target_dir, lang+'.valid')
#             with open(eval_file, mode='w', encoding='utf8') as fp:
#                 for text in texts[train_number:]:
#                     fp.write(text+'\n')
#     print("{} finish".format(data_args.dataset))

# def _extract_chosen_data(
#     file: str,
#     dataset: str,
# ) -> Dict:
#     processed_data: Dict[str,Set] = dict()
#     mention_pattern = re.compile(r'@\S+$|@\S+ ')
#     with open(file, mode='r', encoding='utf8') as fp:
#         for line in fp.readlines():
#             item = json.loads(line.strip())
#             text = item['tweet_text']
#             text = re.sub(mention_pattern, "", text)
#             try:
#                 language = (
#                     item['lang'] if dataset == 'background' else item['tweet_language']
#                 )
#             except KeyError:
#                 print(item.keys())
#             if language in ['', 'und']:
#                 continue
#             if language not in processed_data:
#                 processed_data[language] = set()
#             processed_data[language].add(text)
#     return processed_data


# def _extract_full_data(
#     file:set,
#     misc_args: MiscArgument
# ) -> Dict:
#     processed_data: Dict[str,Set] = dict()
#     count = 0
#     with open(file, encoding='utf8') as tweet_source:
#         tweet_csv = csv.DictReader(tweet_source)
#         for row in tweet_csv:
#             language = row["tweet_language"]
#             if language in ['', 'und']:
#                 continue
#             text = _clean_text(row["tweet_text"])
#             if language not in processed_data:
#                 processed_data[language] = set()
#             processed_data[language].add(text)
#             # count += 1
#             # if misc_args.global_debug and count > 10000:
#             #     break
#     return processed_data


# def _clean_text(original_tweet: str) -> str:
#     processed_tweet = re.sub(r'http[^ ]+', 'URL', original_tweet)
#     processed_tweet = re.sub(r'RT @[^ ]+ ', '', processed_tweet)
#     processed_tweet = re.sub(r'rt @[^ ]+ ', '', processed_tweet)
#     processed_tweet = re.sub(r'@\S+$|@\S+ ','', processed_tweet)
#     processed_tweet = processed_tweet.replace('\n', ' ')
#     processed_tweet = processed_tweet.replace('\r', '')
#     processed_tweet = processed_tweet.replace('RT', '')
#     processed_tweet = processed_tweet.replace('rt', '')
#     processed_tweet = re.sub(r' +', ' ', processed_tweet)
#     processed_tweet = processed_tweet.strip()
#     return processed_tweet


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

def main():
    misc_args, model_args, data_args, training_args, adapter_args, analysis_args = get_config()
    prepare_dirs_and_logger(misc_args, model_args,
                            data_args, training_args, adapter_args,analysis_args)
    # extract_data(misc_args, data_args)
    get_analysis_data(analysis_args)


if __name__ == '__main__':
    main()
