import os
import random
from typing import List
from bs4 import BeautifulSoup
import json
from multiprocessing import Pool
from nltk.tokenize import sent_tokenize
from sklearn.cluster import AgglomerativeClustering
from copy import deepcopy

from .util import prepare_dirs_and_logger
from .config import AnalysisArguments, DataArguments, MiscArgument, get_config, SourceMap, TrustMap, ArticleMap, FullArticleMap



class SelfDataAugmentor(object):
    def __init__(self, misc_args:MiscArgument, data_args:DataArguments) -> None:
        super().__init__()
        self._misc_args = misc_args
        self._data_args = data_args
        self._sequence_length = 256
        self._raw_data = dict()
        self._augmented_data = dict()
        self._article_map = FullArticleMap()

        self._load_original_data()

    def _load_original_data(self):
        self._raw_data = dict()
        media_list = os.listdir(self._data_args.data_dir)
        for media in media_list:
            if media not in self._raw_data:
                self._raw_data[media] = dict()
            grouped_train_data = list()
            grouped_eval_data = list()

            train_file = os.path.join(os.path.join(os.path.join(
                self._data_args.data_dir, media), 'original'), 'en.train')
            eval_file = os.path.join(os.path.join(os.path.join(
                self._data_args.data_dir, media), 'original'), 'en.valid')
            with open(train_file, mode='r', encoding='utf8') as fp:
                for line in fp:
                    paragraph_list = line.strip().split('\\n\\n')
                    for paragraph in paragraph_list:
                        if len(paragraph.split(' '))<self._sequence_length and len(paragraph.split(' '))>5:
                            grouped_train_data.append(paragraph)
                        elif len(paragraph.split(' '))>=self._sequence_length:
                            sentence_list =sent_tokenize(paragraph.strip())
                            chunk_sentences = str()
                            for sentence in sentence_list:
                                if len(chunk_sentences.split(' ')) + len(sentence.split(' ')) < self._sequence_length:
                                    chunk_sentences = chunk_sentences +' '+ sentence
                                else:
                                    grouped_train_data.append(chunk_sentences.strip())
                                    chunk_sentences = sentence
                            grouped_train_data.append(chunk_sentences.strip())
            with open(eval_file, mode='r', encoding='utf8') as fp:
                for line in fp:
                    paragraph_list = line.strip().split('\\n\\n')
                    for paragraph in paragraph_list:
                        if len(paragraph.split(' '))<self._sequence_length and len(paragraph.split(' '))>5:
                            grouped_eval_data.append(paragraph)
                        elif len(paragraph.split(' '))>=self._sequence_length:
                            sentence_list =sent_tokenize(paragraph.strip())
                            chunk_sentences = str()
                            for sentence in sentence_list:
                                if len(chunk_sentences.split(' ')) + len(sentence.split(' ')) < self._sequence_length:
                                    chunk_sentences = chunk_sentences +' '+ sentence
                                else:
                                    grouped_eval_data.append(chunk_sentences.strip())
                                    chunk_sentences = sentence
                            grouped_eval_data.append(chunk_sentences.strip())
            self._raw_data[media] = {'train': grouped_train_data, 'eval': grouped_eval_data}

    def data_augment(self, data_type):
        if data_type == 'sentence_order_replacement':
            self._sentence_order_replacement()

    def _sentence_order_replacement(self):
        for media, media_data in self._raw_data.items():
            if media not in self._augmented_data:
                self._augmented_data[media] = dict()
            train_data = media_data['train']
            eval_data = media_data['eval']

            augmented_train_data = list()
            augmented_eval_data = list()

            for paragraph in train_data:
                sentence_list = sent_tokenize(paragraph.replace(';','.'))
                random.shuffle(sentence_list)
                augmented_sentence_list = deepcopy(sentence_list)
                augmented_train_data.append(augmented_sentence_list)
                if len(sentence_list) > 1:
                    random.shuffle(sentence_list)
                    augmented_sentence_list = deepcopy(augmented_sentence_list)
                    augmented_train_data.append(sentence_list)
            
            augmented_eval_data = eval_data

            self._augmented_data[media]['train'] = augmented_train_data
            self._augmented_data[media]['eval'] = augmented_eval_data
    
    def save(self):
        for media in list(self._augmented_data.keys()):
            data_path = os.path.join(os.path.join(
                self._data_args.data_dir, media), self._data_args.data_type)
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            train_file = os.path.join(data_path, 'en.train')
            random.shuffle(self._augmented_data[media]['train'])
            with open(train_file, mode='w', encoding='utf8') as fp:
                for item in self._augmented_data[media]['train']:
                    fp.write(' '.join(item)+'\n')
            eval_file = os.path.join(data_path, 'en.valid')
            random.shuffle(self._augmented_data[media]['eval'])
            with open(eval_file, mode='w', encoding='utf8') as fp:
                for item in self._augmented_data[media]['eval']:
                    fp.write(' '.join(item)+'\n')




def sentence_order_replacement(
    misc_args: MiscArgument,
    data_args: DataArguments
) -> None:
    article_map = FullArticleMap()
    raw_data = dict()
    train_data = dict()
    eval_data = dict()

    media_list = os.listdir(data_args.data_dir)
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
