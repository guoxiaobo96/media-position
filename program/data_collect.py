from .util import prepare_dirs_and_logger
from .config import AnalysisArguments, DataArguments, MiscArgument, TweetAccount, get_config, SourceMap, TrustMap
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
    cleaned_tweet_list = []

    train_url_list = []
    valid_url_list = []
    train_tweet_list = []
    valid_tweet_list = []
    train_article_list = []
    valid_article_list = []


    tweet_file_path = os.path.join(data_args.data_dir, os.path.join(data_args.dataset, 'twitter'))
    if not os.path.exists(tweet_file_path):
        os.makedirs(tweet_file_path)
    if not os.path.exists(data_args.data_path):
        os.makedirs(data_args.data_path)

    c = twint.Config()
    c.Username = data_args.dataset
    if misc_args.global_debug:
        c.Limit = 100
    c.Store_object = True
    c.Hide_output = True
    twint.run.Search(c)
    tweet_list = twint.output.tweets_list


    for tweet in tweet_list:
        if tweet.lang != 'en':
            continue
        cleaned_tweet_list.append(tweet)
    
    random.seed(123)
    random.shuffle(cleaned_tweet_list)
    train_number = int(len(cleaned_tweet_list)*0.7)


    train_file = os.path.join(tweet_file_path, 'en.train')
    with open(train_file, mode='w', encoding='utf8') as fp:
        for tweet in cleaned_tweet_list[:train_number]:
            text = _clean_text(tweet.tweet)
            fp.write(text+'\n')
            train_url_list.extend(tweet.urls)
    eval_file = os.path.join(tweet_file_path, 'en.valid')
    with open(eval_file, mode='w', encoding='utf8') as fp:
        for tweet in cleaned_tweet_list[train_number:]:
            text = _clean_text(tweet.tweet)
            fp.write(text+'\n')
            valid_url_list.extend(tweet.urls)
    
    for i, url in enumerate(tqdm(train_url_list)):
        try:
            if i!=0 and i%1000==0:
                time.sleep(1)
            content = _download_page(url)
            text_list = _parse_content(content, data_args.dataset)
            if len(text_list) > 5:
                train_article_list.append(text_list)
        except:
            continue

    for i, url in enumerate(tqdm(valid_url_list)):
        try:
            if i!=0 and i%1000==0:
                time.sleep(1)
            content = _download_page(url)
            text_list = _parse_content(content, data_args.dataset)
            if len(text_list) > 5:
                valid_article_list.append(text_list)
        except:
            continue

    train_file = os.path.join(data_args.data_path, 'en.train')
    with open(train_file, mode='w', encoding='utf8') as fp:
        for text_list in train_article_list:
            for text in text_list:
                fp.write(text)
            fp.write('\n')
    eval_file = os.path.join(data_args.data_path, 'en.valid')
    with open(eval_file, mode='w', encoding='utf8') as fp:
        for text_list in valid_article_list:
            for text in text_list:
                fp.write(text)
            fp.write('\n')

    print("{} articles out of {} urls".format(len(train_article_list)+len(valid_article_list), len(train_url_list)+len(valid_url_list)))


def _download_page(url: str) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36',
    }

    r = requests.get(url=url, headers=headers)
    # r.encoding = 'utf-8'
    # r = requests.get(url=url)
    content = r.text
    return content

# parse content
def _parse_content(content: str, dataset: str) -> List[str]:
    if dataset == 'washingtonpost':
        return _washingtonpost_prase_content(content)
    elif dataset == 'nprpolitics':
        return _npr_prase_content(content)
    elif dataset == 'FoxNews':
        return _fox_prase_content(content)
    elif dataset == 'CNN':
        return _cnn_prase_content(content)
    elif dataset in ['MSNBC', 'NBCNews']:
        return _msnbbc_and_nbcnews_prase_content(content)
    else:
        return _basic_prase_content(content)


def _basic_prase_content(content: str) -> List[str]:
    text_list = []
    soup = BeautifulSoup(content, 'html.parser')
    para_list = soup.find_all('p')
    for para in para_list:
        text_list.append(" ".join(para.text.split()))
    return text_list


def _washingtonpost_prase_content(content: str) -> List[str]:
    text_list = []
    soup = BeautifulSoup(content, 'html.parser')
    para_list = soup.find_all('p')
    for para in para_list:
        if not para.attrs or 'data-el' in para.attrs:
            text_list.append(" ".join(para.text.split()))
    return text_list


def _npr_prase_content(content: str) -> List[str]:
    text_list = []
    soup = BeautifulSoup(content, 'html.parser')
    para_list = soup.find_all('p')
    for para in para_list:
        if not para.attrs or ('class' in para.attrs and not 'left' in para.attrs['class'] and not 'right' in para.attrs['class']):
            text_list.append(" ".join(para.text.split()))
    return text_list


def _fox_prase_content(content: str) -> List[str]:
    text_list = []
    soup = BeautifulSoup(content, 'html.parser')
    para_list = soup.find_all('p')
    for para in para_list:
        if not para.attrs or ('class' in para.attrs and 'speakable' in para.attrs['class']):
            text_list.append(" ".join(para.text.split()))
    return text_list


def _cnn_prase_content(content: str) -> List[str]:
    text_list = []
    soup = BeautifulSoup(content, 'html.parser')
    para_list = soup.find_all('h1')
    for para in para_list:
        text_list.append(" ".join(para.text.split()))
    para_list = soup.find_all('div')
    for para in para_list:
        if 'class' in para.attrs:
            for tag in para.attrs['class']:
                if 'paragraph' in tag or 'Paragraph' in tag:
                    text_list.append(" ".join(para.text.split()))
    para_list = soup.find_all('p')
    for para in para_list:
        if 'class' in para.attrs:
            for tag in para.attrs['class']:
                if 'paragraph' in tag or 'Paragraph' in tag:
                    text_list.append(" ".join(para.text.split()))
    return text_list


def _msnbbc_and_nbcnews_prase_content(content: str) -> List[str]:
    text_list = []
    soup = BeautifulSoup(content, 'html.parser')
    para_list = soup.find_all('p')
    for para in para_list:
        if not para.attrs or ('class' in para.attrs and not 'menu-section-heading' in para.attrs['class']):
            text_list.append(" ".join(para.text.split()))
    return text_list


def main():
    misc_args, model_args, data_args, training_args, adapter_args, analysis_args = get_config()
    prepare_dirs_and_logger(misc_args, model_args,
                            data_args, training_args, adapter_args, analysis_args)
    twitter_collect(misc_args, data_args)


if __name__ == '__main__':
    main()
