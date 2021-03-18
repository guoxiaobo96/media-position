from multiprocessing.pool import ApplyResult
from .util import prepare_dirs_and_logger
from .config import AnalysisArguments, DataArguments, MiscArgument, TweetAccount,get_config, SourceMap, TrustMap
import tweepy
import time
import os
import sys
from os import path
import warnings
import json
import csv
import re
from multiprocessing import Pool
import random
from typing import Any, List, Optional, Union, Dict, Set, List
from glob import glob
warnings.filterwarnings('ignore')
import twint

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
        if len(tweet_list)<1000:
            continue
        random.seed(123)
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


def _limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            time.sleep(15 * 60)
        except StopIteration:
            break

def _clean_text(original_tweet: str) -> str:
    processed_tweet = re.sub(r'http[^ ]+', 'URL', original_tweet)
    processed_tweet = re.sub(r'RT @[^ ]+ ', '', processed_tweet)
    processed_tweet = re.sub(r'rt @[^ ]+ ', '', processed_tweet)
    processed_tweet = re.sub(r'@\S+$|@\S+ ','', processed_tweet)
    processed_tweet = processed_tweet.replace('\n', ' ')
    processed_tweet = processed_tweet.replace('\r', '')
    processed_tweet = processed_tweet.replace('RT', '')
    processed_tweet = processed_tweet.replace('rt', '')
    processed_tweet = re.sub(r' +', ' ', processed_tweet)
    processed_tweet = processed_tweet.strip()
    return processed_tweet

def main():
    misc_args, model_args, data_args, training_args, adapter_args, analysis_args = get_config()
    prepare_dirs_and_logger(misc_args, model_args,
                            data_args, training_args, adapter_args, analysis_args)
    twitter_collect(misc_args, data_args)

if __name__ =='__main__':
    main()