from genericpath import exists
import os
import random
from typing import List
from bs4 import BeautifulSoup
import json
from multiprocessing import Pool
from nltk.tokenize import sent_tokenize
from sklearn.cluster import AgglomerativeClustering
from copy import copy, deepcopy
import torch
import tqdm
from gensim.models import Word2Vec
from nltk.corpus import stopwords

from .util import prepare_dirs_and_logger
from .config import AnalysisArguments, DataArguments, MiscArgument, get_config, SourceMap, TrustMap, ArticleMap, FullArticleMap, DataAugArguments


class SelfDataAugmentor(object):
    def __init__(self, misc_args: MiscArgument, data_args: DataArguments, aug_args: DataAugArguments) -> None:
        super().__init__()
        self._misc_args = misc_args
        self._data_args = data_args
        self._aug_args = aug_args
        self._sequence_length = 256
        self._data_augmentor = DataAugmentor()
        self._raw_data = dict()
        self._augmented_data = dict()
        self._article_map = FullArticleMap()
        self._augment_method_map = {'duplicate': self._duplicate, 'no_augmentation': self._no_augmentation, 'sentence_order_replacement': self._sentence_order_replacement,
                                    'span_cutoff': self._span_cutoff, 'word_order_replacement': self._word_order_replacement, 'word_replacement': self._word_replacement}

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
                        if len(paragraph.split(' ')) < self._sequence_length and len(paragraph.split(' ')) > 5:
                            grouped_train_data.append(paragraph)
                        elif len(paragraph.split(' ')) >= self._sequence_length:
                            sentence_list = sent_tokenize(paragraph.strip())
                            chunk_sentences = str()
                            for sentence in sentence_list:
                                if len(chunk_sentences.split(' ')) + len(sentence.split(' ')) < self._sequence_length:
                                    chunk_sentences = chunk_sentences + ' ' + sentence
                                else:
                                    grouped_train_data.append(
                                        chunk_sentences.strip())
                                    chunk_sentences = sentence
                            grouped_train_data.append(chunk_sentences.strip())
            with open(eval_file, mode='r', encoding='utf8') as fp:
                for line in fp:
                    paragraph_list = line.strip().split('\\n\\n')
                    for paragraph in paragraph_list:
                        if len(paragraph.split(' ')) < self._sequence_length and len(paragraph.split(' ')) > 5:
                            grouped_eval_data.append(paragraph)
                        elif len(paragraph.split(' ')) >= self._sequence_length:
                            sentence_list = sent_tokenize(paragraph.strip())
                            chunk_sentences = str()
                            for sentence in sentence_list:
                                if len(chunk_sentences.split(' ')) + len(sentence.split(' ')) < self._sequence_length:
                                    chunk_sentences = chunk_sentences + ' ' + sentence
                                else:
                                    grouped_eval_data.append(
                                        chunk_sentences.strip())
                                    chunk_sentences = sentence

                            grouped_eval_data.append(chunk_sentences.strip())
            self._raw_data[media] = {
                'train': grouped_train_data, 'eval': grouped_eval_data}

    def data_augment(self, augment_type):
        self._augment_method = self._augment_method_map[augment_type]

        for media, media_data in tqdm.tqdm(self._raw_data.items()):
            if media not in self._augmented_data:
                self._augmented_data[media] = dict()
            train_data = media_data['train']
            eval_data = media_data['eval']

            augmented_train_data = list()
            augmented_eval_data = list()

            if augment_type in ['word_replacement']:
                sentence_list = [s.split(' ') for s in train_data]
                model = Word2Vec(sentences=sentence_list,
                                window=5, min_count=1, workers=4)
            else:
                model = None

            for index, paragraph in enumerate(train_data):
                item = {'original': paragraph, 'augmented': list()}

                augmented_sentence = self._augment_method(paragraph, model)
                item['augmented']=augmented_sentence
                augmented_train_data.append(item)

                if self._misc_args.global_debug and index>100:
                    break

            augmented_eval_data = eval_data

            self._augmented_data[media]['train'] = augmented_train_data
            self._augmented_data[media]['eval'] = augmented_eval_data

    def _no_augmentation(self, paragraph, model):
        augmented_data = list()
        return augmented_data

    def _duplicate(self, paragraph, model):
        augmented_data = list()

        for _ in range(self._aug_args.multiple_number - 1):
            augmented_data.append(paragraph)
        
        return augmented_data

    def _sentence_order_replacement(self, paragraph, model):
        existing_data = [paragraph]
        augmented_data = list()

        for _ in range(self._aug_args.multiple_number - 1):
            augmented_sentence = self._data_augmentor.sentence_order_replacement(
                paragraph)
            count = 0
            while augmented_sentence in augmented_data:
                count += 1
                augmented_sentence = self._data_augmentor.sentence_order_replacement(
                    paragraph)
                if count > 3:
                    break
            augmented_data.append(augmented_sentence)
            existing_data.append(augmented_sentence)

        return augmented_data

    def _span_cutoff(self, paragraph, model):
        existing_data = [paragraph]
        augmented_data = list()
        num_span = max(1, int(0.1*len(paragraph.split(' '))))

        for _ in range(self._aug_args.multiple_number - 1):
            augmented_sentence = self._data_augmentor.span_cutoff(
                paragraph, num_span)
            count = 0
            while augmented_sentence in augmented_data:
                count += 1
                augmented_sentence = self._data_augmentor.span_cutoff(
                    paragraph, num_span)
                if count > 3:
                    break
            augmented_data.append(augmented_sentence)
            existing_data.append(augmented_sentence)
        return augmented_data

    def _word_order_replacement(self, paragraph, model):
        existing_data = [paragraph]
        augmented_data = list()
        num_swap = max(1, int(0.1*len(paragraph.split(' '))))

        for _ in range(self._aug_args.multiple_number - 1):
            augmented_sentence = self._data_augmentor.word_order_replacement(
                paragraph, num_swap)
            count = 0
            while augmented_sentence in augmented_data:
                count += 1
                augmented_sentence = self._data_augmentor.word_order_replacement(
                    paragraph, num_swap)
                if count > 3:
                    break
            augmented_data.append(augmented_sentence)
            existing_data.append(augmented_sentence)

        return augmented_data

    def _word_replacement(self, paragraph, model):
        existing_data = [paragraph]
        augmented_data = list()
        num_replacement = max(1, int(0.1*len(paragraph.split(' '))))
        
        for _ in range(self._aug_args.multiple_number - 1):
            augmented_sentence = self._data_augmentor.word_replacement(
                paragraph, model, num_replacement)
            count = 0
            while augmented_sentence in augmented_data:
                count += 1
                augmented_sentence = self._data_augmentor.word_replacement(
                    paragraph, model, num_replacement)
                if count > 3:
                    break
            augmented_data.append(augmented_sentence)
            existing_data.append(augmented_sentence)
        return augmented_data


    def _back_translation(self):
        en2de = torch.hub.load(
            'pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
        de2en = torch.hub.load(
            'pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

        en2de.cuda()
        de2en.cuda()

        for media, media_data in tqdm.tqdm(self._raw_data.items()):
            if media not in self._augmented_data:
                self._augmented_data[media] = dict()
            train_data = media_data['train']
            eval_data = media_data['eval']

            augmented_train_data = list()
            augmented_eval_data = list()

            augmented_train_data.extend(train_data)
            augmented_data = de2en.translate(en2de.translate(train_data))
            augmented_train_data.extend(augmented_data)

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
            with open(train_file, mode='w', encoding='utf8') as fp:
                for item in self._augmented_data[media]['train']:
                    fp.write(json.dumps(item, ensure_ascii=False)+'\n')
            eval_file = os.path.join(data_path, 'en.valid')
            random.shuffle(self._augmented_data[media]['eval'])
            with open(eval_file, mode='w', encoding='utf8') as fp:
                for item in self._augmented_data[media]['eval']:
                    fp.write(item+'\n')


class DataAugmentor(object):
    def __init__(self) -> None:
        super().__init__()

    def sentence_order_replacement(self, paragraph):
        sentence_list = sent_tokenize(paragraph.replace(';', '.'))
        random.shuffle(sentence_list)
        augmented_sentence_list = deepcopy(sentence_list)
        augmented_sentence = ' '.join(augmented_sentence_list)
        return augmented_sentence

    def span_cutoff(self, paragraph, num_span):
        splited_paragraph = paragraph.split(' ')
        length = len(splited_paragraph)

        start_index = random.randint(0, length-num_span)
        cutoff_paragraph = splited_paragraph[:start_index] + \
            splited_paragraph[start_index+num_span:]
        cutoff_paragraph = ' '.join(cutoff_paragraph)

        return cutoff_paragraph

    def word_replacement(self, paragraph, model, num_replacement):
        original_splited_paragraph = paragraph.split(' ')
        length = len(original_splited_paragraph)
        for _ in range(num_replacement):
            splited_paragraph = deepcopy(original_splited_paragraph)
            replace_position = random.randint(0, length-1)
            replaced_word = splited_paragraph[replace_position]
            while replaced_word in stopwords.words('english'):
                replace_position = random.randint(0, length-1)
                replaced_word = splited_paragraph[replace_position]
            splited_paragraph[replace_position] = model.wv.most_similar(
                replaced_word, topn=1)[0][0]
        return ' '.join(splited_paragraph)

    def word_order_replacement(self, paragraph, num_swap):
        length = len(paragraph.split(' '))
        splited_paragraph = paragraph.split(' ')

        for i in range(num_swap):
            random_idx_1 = random.randint(0, length-1)
            random_idx_2 = random.randint(0, length-1)
            counter = 0
            while random_idx_2 == random_idx_1:
                random_idx_2 = random.randint(0, length-1)
                counter += 1
                if counter > 3:
                    break
            splited_paragraph[random_idx_1], splited_paragraph[random_idx_2] = splited_paragraph[random_idx_2], splited_paragraph[random_idx_1]
        augmented_sentence = ' '.join(splited_paragraph)

        return augmented_sentence
