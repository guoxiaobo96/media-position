from torch import mode
from .util import prepare_dirs_and_logger
from .config import AnalysisArguments, DataArguments, MiscArgument, get_config, SourceMap, TrustMap, ArticleMap
import tweepy
import time
import os
import warnings
import re
import random
from typing import List
import twint
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import json
from multiprocessing import Pool

warnings.filterwarnings('ignore')


def twitter_collect(
    misc_args: MiscArgument,
    data_args: DataArguments
) -> None:
    if not os.path.exists(data_args.data_path):
        os.makedirs(data_args.data_path)
    tweet_list = []
    cleaned_tweet_dict = dict()

    # tweet_account = TweetAccount()
    # auth = tweepy.OAuthHandler(tweet_account.consumer_key, tweet_account.consumer_secret)
    # # auth.set_access_token(tweet_account.access_token, tweet_account.access_token_secret)
    # api = tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    # max_id = 1369413687413415936
    # # max_id = 1359189065338290176

    # for page in _limit_handled(tweepy.Cursor(api.user_timeline, id=data_args.dataset,tweet_mode="extended", max_id=max_id).pages()):
    #     # page is a list of statuses
    #     tweet_list.extend(page)
    #     max_id = page[-1].id

    c = twint.Config()
    c.Username = data_args.dataset
    c.Custom["tweet"] = ["tweet"]
    if misc_args.global_debug:
        c.Limit = 10000
    c.Store_object = True
    c.Hide_output = True
    twint.run.Search(c)
    tweet_list = twint.output.tweets_list

    for tweet in tweet_list:
        if tweet.lang in ['', 'und']:
            continue
        if tweet.lang not in cleaned_tweet_dict:
            cleaned_tweet_dict[tweet.lang] = list()
        cleaned_tweet_dict[tweet.lang].append(_clean_text(tweet.tweet))

    for lang, tweet_list in cleaned_tweet_dict.items():
        if len(tweet_list) < 1000:
            continue
        random.shuffle(tweet_list)
        train_number = int(len(tweet_list)*0.7)
        train_file = os.path.join(data_args.data_path, lang+'.train')
        with open(train_file, mode='w', encoding='utf8') as fp:
            for text in tweet_list[:train_number]:
                fp.write(text+'\n')
        eval_file = os.path.join(data_args.data_path, lang+'.valid')
        with open(eval_file, mode='w', encoding='utf8') as fp:
            for text in tweet_list[train_number:]:
                fp.write(text+'\n')


def _clean_text(original_tweet: str) -> str:
    processed_tweet = re.sub(r'http[^ ]+', 'URL', original_tweet)
    processed_tweet = re.sub(r'RT @[^ ]+ ', '', processed_tweet)
    processed_tweet = re.sub(r'rt @[^ ]+ ', '', processed_tweet)
    processed_tweet = re.sub(r'@\S+$|@\S+ ', '', processed_tweet)
    processed_tweet = processed_tweet.replace('\n', ' ')
    processed_tweet = processed_tweet.replace('\r', '')
    processed_tweet = processed_tweet.replace('RT', '')
    processed_tweet = processed_tweet.replace('rt', '')
    processed_tweet = re.sub(r' +', ' ', processed_tweet)
    processed_tweet = processed_tweet.strip()
    return processed_tweet


def article_collect(
    misc_args: MiscArgument,
    data_args: DataArguments
) -> None:  
    article_map = ArticleMap()

    article_dict = dict()
    data_path_dir_list_temp = []
    data_path_dir_list = []
    file_path_list = []

    year_list = os.listdir(data_args.original_data_dir)
    for year in year_list:
        data_path_dir = os.path.join(data_args.original_data_dir, year)
        data_path_dir_list_temp.append(data_path_dir)

    for data_path_year in data_path_dir_list_temp:
        # topic_list = os.listdir(data_path_year)
        topic_list = ['obamacare']
        for topic in topic_list:
            data_path_dir = os.path.join(data_path_year, topic)
            data_path_dir_list.append(data_path_dir)
    
    result_list = []
    if misc_args.global_debug:
        for data_path_dir in data_path_dir_list:
            article_dict_temp = _article_collect(data_path_dir, misc_args.global_debug)
            result_list.append(article_dict_temp)
    else:
        with Pool(processes=10) as pool:
            for data_path_dir in data_path_dir_list:
                article_dict_temp = pool.apply_async(func=_article_collect, args=(data_path_dir, misc_args.global_debug,))
                result_list.append(article_dict_temp)
            pool.close()
            pool.join()

    for result in result_list:
        if misc_args.global_debug:
            result = result
        else:
            result = result.get()
        for media, text_list in result.items():
            if media not in article_dict:
                article_dict[media] = list()
            article_dict[media].extend(text_list)

    for media_name, text_list in article_dict.items():
        if media_name not in article_map.name_to_dataset:
            continue
        media = article_map.name_to_dataset[media_name]
        text_list = list(set(text_list))
        random.shuffle(text_list)
        train_number = int(len(text_list)*0.7)
        data_path = os.path.join(os.path.join(data_args.data_dir, media),data_args.data_type)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        train_file = os.path.join(data_path, 'en.train')
        with open(train_file, mode='w', encoding='utf8') as fp:
            for text in text_list[:train_number]:
                if len(text)<5:
                    continue
                if len(text) > 512:
                    start_index = 0
                    end_index = 0
                    while start_index < len(text):
                        text_t = text[start_index:start_index+512]
                        end_index = text_t.rfind('.')
                        if end_index != -1:
                            text_t = text_t[:end_index+1]
                        text_t = text_t.strip()
                        fp.write(text_t+'\n')
                        if end_index != -1:
                            start_index += (end_index+1)
                        else:
                            start_index +=512
                else:
                    fp.write(text+'\n')
        eval_file = os.path.join(data_path, 'en.valid')
        with open(eval_file, mode='w', encoding='utf8') as fp:
            for text in text_list[train_number:]:
                if len(text)<5:
                    continue
                if len(text) > 512:
                    start_index = 0
                    end_index = 0
                    while start_index < len(text):
                        text_t = text[start_index:start_index+512]
                        end_index = text_t.rfind('.')
                        if end_index != -1:
                            text_t = text_t[:end_index+1]
                        text_t.strip()
                        fp.write(text_t+'\n')
                        if end_index != -1:
                            start_index += (end_index+1)
                        else:
                            start_index +=512
                else:
                    fp.write(text+'\n')

def _article_collect(data_path_dir, global_debug):
    article_dict = dict()
    file_path_list = os.listdir(data_path_dir)
    if global_debug:
        file_path_list = file_path_list[:30]
    for file in file_path_list:
        file_path = os.path.join(data_path_dir, file)
        with open(file_path, mode='r',encoding='utf8') as fp:
            item = json.load(fp)
            text = item['text']
            media = item['media']
        if media not in article_dict:
            article_dict[media] = list()
        article_dict[media].append(_clean_text(text))
    return article_dict

def main():
    misc_args, model_args, data_args, training_args, adapter_args, analysis_args = get_config()
    prepare_dirs_and_logger(misc_args, model_args,
                            data_args, training_args, adapter_args, analysis_args)
    # twitter_collect(misc_args, data_args)
    article_collect(misc_args, data_args)


if __name__ == '__main__':
    main()
