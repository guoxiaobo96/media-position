from platform import node
from numpy.core import einsumfunc
from torch import chunk, dist, mode
from .util import prepare_dirs_and_logger
from .config import AnalysisArguments, DataArguments, MiscArgument, get_config, SourceMap, TrustMap, ArticleMap, FullArticleMap
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
from nltk.tokenize import sent_tokenize
from sklearn.cluster import AgglomerativeClustering
import csv

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

    train_topic_list = ['abortion', 'marijuana', 'drug policy', 'gay marriage']
    eval_topic_list = ['corporate tax']
    test_topic_list = ['climate change', 'obamacare']

    topic_dict = {'train': train_topic_list,
                  'eval': eval_topic_list, 'test': test_topic_list}

    if data_args.dataset in topic_dict:
        topic_list = topic_dict[data_args.dataset]
    else:
        topic_list = [data_args.dataset]
    data_args.data_dir = data_args.data_dir + '_' + data_args.dataset

    year_list = os.listdir(data_args.original_data_dir)
    for year in year_list:
        data_path_dir = os.path.join(data_args.original_data_dir, year)
        data_path_dir_list_temp.append(data_path_dir)

    for data_path_year in data_path_dir_list_temp:
        # topic_list = os.listdir(data_path_year)
        for topic in topic_list:
            data_path_dir = os.path.join(data_path_year, topic)
            data_path_dir_list.append(data_path_dir)

    result_list = []
    if misc_args.global_debug:
        for data_path_dir in data_path_dir_list:
            article_dict_temp = _article_collect(
                data_path_dir, misc_args.global_debug)
            result_list.append(article_dict_temp)
    else:
        with Pool(processes=10) as pool:
            for data_path_dir in data_path_dir_list:
                article_dict_temp = pool.apply_async(
                    func=_article_collect, args=(data_path_dir, misc_args.global_debug,))
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
        data_path = os.path.join(os.path.join(
            data_args.data_dir, media), data_args.data_type)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        train_file = os.path.join(data_path, 'en.train')
        with open(train_file, mode='w', encoding='utf8') as fp:
            for text in text_list[:train_number]:
                if len(text) < 5:
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
                            start_index += 512
                else:
                    fp.write(text+'\n')
        eval_file = os.path.join(data_path, 'en.valid')
        with open(eval_file, mode='w', encoding='utf8') as fp:
            for text in text_list[train_number:]:
                if len(text) < 5:
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
                            start_index += 512
                else:
                    fp.write(text+'\n')


def _article_collect(data_path_dir, global_debug):
    article_dict = dict()
    file_path_list = os.listdir(data_path_dir)
    if global_debug:
        file_path_list = file_path_list[:30]
    for file in file_path_list:
        file_path = os.path.join(data_path_dir, file)
        with open(file_path, mode='r', encoding='utf8') as fp:
            item = json.load(fp)
            text = item['text']
            media = item['media']
        if media not in article_dict:
            article_dict[media] = list()
        article_dict[media].append(_clean_text(text))
    return article_dict


def origianl_collect(
    misc_args: MiscArgument,
    data_args: DataArguments
) -> None:
    article_map = FullArticleMap()

    article_dict = dict()
    data_path_dir_list_temp = []
    data_path_dir_list = []

    train_topic_list = ['abortion', 'marijuana', 'drug policy', 'gay marriage']
    eval_topic_list = ['corporate tax']
    test_topic_list = ['climate change', 'obamacare']

    topic_dict = {'train': train_topic_list,
                  'eval': eval_topic_list, 'test': test_topic_list}

    if data_args.dataset in topic_dict:
        topic_list = topic_dict[data_args.dataset]
    else:
        topic_list = [data_args.dataset.replace('-',' ')]
    data_args.data_dir = data_args.data_dir + '_' + data_args.dataset

    year_list = os.listdir(data_args.original_data_dir)
    for year in year_list:
        data_path_dir = os.path.join(data_args.original_data_dir, year)
        data_path_dir_list_temp.append(data_path_dir)

    for data_path_year in data_path_dir_list_temp:
        for topic in topic_list:
            data_path_dir = os.path.join(data_path_year, topic)
            data_path_dir_list.append(data_path_dir)

    result_list = []
    if misc_args.global_debug:
        for data_path_dir in data_path_dir_list:
            article_dict_temp = _original_collect(
                data_path_dir, misc_args.global_debug)
            result_list.append(article_dict_temp)
    else:
        with Pool(processes=10) as pool:
            for data_path_dir in data_path_dir_list:
                article_dict_temp = pool.apply_async(
                    func=_original_collect, args=(data_path_dir, misc_args.global_debug,))
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
        if len(text_list) < 50:
            continue
        random.shuffle(text_list)
        train_number = int(len(text_list)*0.9)
        data_path = os.path.join(os.path.join(
            data_args.data_dir, media), data_args.data_type)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        train_file = os.path.join(data_path, 'en.train')
        with open(train_file, mode='w', encoding='utf8') as fp:
            for text in text_list[:train_number]:
                if len(text) < 5:
                    continue
                fp.write(text+'\n')
        eval_file = os.path.join(data_path, 'en.valid')
        with open(eval_file, mode='w', encoding='utf8') as fp:
            for text in text_list[train_number:]:
                if len(text) < 5:
                    continue
                fp.write(text+'\n')


def _original_collect(data_path_dir, global_debug):
    article_dict = dict()
    file_path_list = os.listdir(data_path_dir)
    if global_debug:
        file_path_list = file_path_list[:30]
    for file in file_path_list:
        file_path = os.path.join(data_path_dir, file)
        with open(file_path, mode='r', encoding='utf8') as fp:
            item = json.load(fp)
            text = item['text']
            media = item['media']
        if media not in article_dict:
            article_dict[media] = list()
        text = text.strip().replace('\n','\\n').replace('\"','')
        text = text.lower()
        article_dict[media].append(text)
    return article_dict


def sentence_random_replacement_collect(
    misc_args: MiscArgument,
    data_args: DataArguments
) -> None:
    article_map = FullArticleMap()
    raw_data = dict()
    train_data = dict()
    eval_data = dict()

    media_list = os.listdir(data_args.data_dir)
    distance_dict = _calcualte_score('article', 'trust', media_list)
    for media in media_list:
        if media not in raw_data:
            raw_data[media] = dict()
        splited_train_data = list()
        splited_eval_data = list()
        train_sentences_list = list()
        eval_sentences_list = list()

        train_file = os.path.join(os.path.join(os.path.join(
            data_args.data_dir, media), 'original'), 'en.train')
        eval_file = os.path.join(os.path.join(os.path.join(
            data_args.data_dir, media), 'original'), 'en.valid')
        with open(train_file, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                sentences_list = sent_tokenize(line.strip())
                train_sentences_list.extend(sentences_list)
                grouped_sentences = list()
                for i in range(len(sentences_list) - 2):
                    grouped_sentences.append(
                        [sentences_list[i], sentences_list[i+1], sentences_list[i+2]])
                splited_train_data.extend(grouped_sentences)
        with open(eval_file, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                sentences_list = sent_tokenize(line.strip())
                eval_sentences_list.extend(sentences_list)
                grouped_sentences = list()
                for i in range(len(sentences_list) - 2):
                    grouped_sentences.append(
                        [sentences_list[i], sentences_list[i+1], sentences_list[i+2]])
                splited_eval_data.extend(grouped_sentences)
        raw_data[media] = {'train_sentences_list': train_sentences_list, 'eval_sentences_list': eval_sentences_list,
                           'split_train_data': splited_train_data, 'split_eval_data': splited_eval_data}

    for media in media_list:
        if media not in train_data:
            train_data[media] = list()
        if media not in eval_data:
            eval_data[media] = list()

        for sentence_list in raw_data[media]['split_train_data']:
            for replace_media in media_list:
                replaced_sentence = random.choice(raw_data[replace_media]['train_sentences_list'])
                while sentence_list[1] == replaced_sentence:
                    replaced_sentence = random.choice(raw_data[replace_media]['train_sentences_list'])
                sentence = sentence_list[0] + replaced_sentence + sentence_list[2]
                train_data[media].append({'sentence':sentence,'label':distance_dict[media][replace_media]})
        for sentence_list in raw_data[media]['split_eval_data']:
            for replace_media in media_list:
                replaced_sentence = random.choice(raw_data[replace_media]['eval_sentences_list'])
                while sentence_list[1] == replaced_sentence:
                    replaced_sentence = random.choice(raw_data[replace_media]['eval_sentences_list'])
                sentence = sentence_list[0] + replaced_sentence + sentence_list[2]
                eval_data[media].append({'sentence':sentence,'label':distance_dict[media][replace_media]})
    for media in media_list:
        data_path = os.path.join(os.path.join(
            data_args.data_dir, media), data_args.data_type)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        train_file = os.path.join(data_path, 'en.train')
        random.shuffle(train_data[media])
        with open(train_file, mode='w', encoding='utf8') as fp:
            for item in train_data[media]:
                fp.write(json.dumps(item, ensure_ascii=False)+'\n')
        eval_file = os.path.join(data_path, 'en.valid')
        random.shuffle(eval_data[media])
        with open(eval_file, mode='w', encoding='utf8') as fp:
            for item in eval_data[media]:
                fp.write(json.dumps(item, ensure_ascii=False)+'\n')


def paragraph_collect(
    misc_args: MiscArgument,
    data_args: DataArguments
) -> None:
    sequence_length = 256
    article_map = FullArticleMap()
    raw_data = dict()
    train_data = dict()
    eval_data = dict()

    media_list = os.listdir(data_args.data_dir)
    for media in media_list:
        if media not in raw_data:
            raw_data[media] = dict()
        grouped_train_data = list()
        grouped_eval_data = list()

        train_file = os.path.join(os.path.join(os.path.join(
            data_args.data_dir, media), 'original'), 'en.train')
        eval_file = os.path.join(os.path.join(os.path.join(
            data_args.data_dir, media), 'original'), 'en.valid')
        with open(train_file, mode='r', encoding='utf8') as fp:
            for line in fp:
                paragraph_list = line.strip().split('\\n\\n')
                for paragraph in paragraph_list:
                    if len(paragraph.split(' '))<sequence_length and len(paragraph.split(' '))>5:
                        grouped_train_data.append(paragraph)
                    elif len(paragraph.split(' '))>=sequence_length:
                        sentence_list =sent_tokenize(paragraph.strip())
                        chunk_sentences = str()
                        for sentence in sentence_list:
                            if len(chunk_sentences.split(' ')) + len(sentence.split(' ')) < sequence_length:
                                chunk_sentences = chunk_sentences +' '+ sentence
                            else:
                                grouped_train_data.append(chunk_sentences.strip())
                                chunk_sentences = sentence
                        grouped_train_data.append(chunk_sentences.strip())
        with open(eval_file, mode='r', encoding='utf8') as fp:
            for line in fp:
                paragraph_list = line.strip().split('\\n\\n')
                for paragraph in paragraph_list:
                    if len(paragraph.split(' '))<sequence_length and len(paragraph.split(' '))>5:
                        grouped_eval_data.append(paragraph)
                    elif len(paragraph.split(' '))>=sequence_length:
                        sentence_list =sent_tokenize(paragraph.strip())
                        chunk_sentences = str()
                        for sentence in sentence_list:
                            if len(chunk_sentences.split(' ')) + len(sentence.split(' ')) < sequence_length:
                                chunk_sentences = chunk_sentences +' '+ sentence
                            else:
                                grouped_eval_data.append(chunk_sentences.strip())
                                chunk_sentences = sentence
                        grouped_eval_data.append(chunk_sentences.strip())
        raw_data[media] = {'train': grouped_train_data, 'eval': grouped_eval_data}


    for media in media_list:
        data_path = os.path.join(os.path.join(
            data_args.data_dir, media), data_args.data_type)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        train_file = os.path.join(data_path, 'en.train')
        random.shuffle(raw_data[media]['train'])
        with open(train_file, mode='w', encoding='utf8') as fp:
            for item in raw_data[media]['train']:
                fp.write(item+'\n')
        eval_file = os.path.join(data_path, 'en.valid')
        random.shuffle(raw_data[media]['eval'])
        with open(eval_file, mode='w', encoding='utf8') as fp:
            for item in raw_data[media]['eval']:
                fp.write(item+'\n')

                
def _calcualte_score(data_type, label_type, media_list):
    label_list = media_list
    article_map = FullArticleMap()

    data = list()
    data_temp = dict()
    with open('./analysis/baseline/baseline_'+label_type+'_'+data_type+'.csv', mode='r', encoding='utf8') as fp:
        reader = csv.reader(fp)
        header = next(reader)
        for row in reader:
            data_temp[row[0]] = [float(x.strip()) for x in row[1:]]
    for k in label_list:
        data.append(data_temp[article_map.dataset_to_name[k]])
    analyzer = AgglomerativeClustering(
        n_clusters=2, compute_distances=True, affinity='cosine', linkage='single')
    clusters = analyzer.fit(data)

    child_node = clusters.children_

    inter_node_dict = dict()
    for i, merge in enumerate(child_node):
        node_index = i+clusters.n_leaves_
        if node_index not in inter_node_dict:
            inter_node_dict[node_index] = dict()
        for child_idx in merge:
            if child_idx in inter_node_dict:
                for k, v in inter_node_dict[child_idx].items():
                    inter_node_dict[node_index][k] = v+1
            else:
                inter_node_dict[node_index][child_idx] = 1

    distance_dict = dict()
    for i, a_name in enumerate(media_list):
        if a_name not in distance_dict:
            distance_dict[a_name] = dict()
        for j, b_name in enumerate(media_list):
            if i==j:
                distance_dict[a_name][b_name] = 0
                continue
            max_distance = 1000
            for k, node_list in inter_node_dict.items():
                if i in node_list and j in node_list:
                    max_distance = min(max_distance,max(node_list[i],node_list[j])+len(node_list)-2)
                    # max_distance = min(max_distance,max(node_list[i],node_list[j])+node_list[i]+node_list[j])
            distance_dict[a_name][b_name] = max_distance   


    # distance_matrix = [[1000 for _ in range(len(label_list)*2 - 1)] for _ in range(len(label_list)*2 - 1)]
    # child_node = clusters.children_
    # for i, merge in enumerate(child_node):
    #     for child_idx in merge:
    #         distance_matrix[child_idx][i+clusters.n_leaves_] = 1
    #         distance_matrix[i+clusters.n_leaves_][child_idx] = 1
    # for k in range(len(label_list)*2 - 1):
    #     distance_matrix[k][k] = 0
    # for k in range(len(label_list)*2 - 1):
    #     for i in range(len(label_list)*2 - 1):
    #         for j in range(len(label_list)*2 - 1):
    #             distance_matrix[i][j] = min(distance_matrix[i][j], distance_matrix[i][k]+distance_matrix[k][j])
    # distance_dict = dict()
    # for i, a_name in enumerate(media_list):
    #     if a_name not in distance_dict:
    #         distance_dict[a_name] = dict()
    #     for j, b_name in enumerate(media_list):
    #         distance_dict[a_name][b_name] = distance_matrix[i][j]

    for k,dist in distance_dict.items():
        new_dist = dict()
        adj_distance_list = sorted(set(dist.values()))
        for media, d in dist.items():
            new_dist[media] = adj_distance_list.index(d)
        distance_dict[k] = new_dist


    return distance_dict
    


def data_collect(
    misc_args: MiscArgument,
    data_args: DataArguments
) -> None:
    if data_args.data_type == 'original':
        origianl_collect(misc_args, data_args)
    elif data_args.data_type == 'sentence_random_replacement':
        sentence_random_replacement_collect(misc_args, data_args)
    elif data_args.data_type in ['mlm', 'lm']:
        paragraph_collect(misc_args, data_args)


def main():
    misc_args, model_args, data_args, training_args, adapter_args, analysis_args = get_config()
    prepare_dirs_and_logger(misc_args, model_args,
                            data_args, training_args, adapter_args, analysis_args)
    # twitter_collect(misc_args, data_args)
    # article_collect(misc_args, data_args)
    data_collect(misc_args, data_args)


if __name__ == '__main__':
    main()
