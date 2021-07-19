import os
from typing import Dict
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np

from .config import MiscArgument, BaselineArguments, DataArguments


class BaselineCalculator(object):
    def __init__(
        self, 
        misc_args:MiscArgument, 
        baseline_args:BaselineArguments, 
        data_args:DataArguments
    ) -> None:
        super().__init__()
        self._misc_args = misc_args
        self._baseline_args = baseline_args
        self._data_args = data_args
        self._original_media_data = dict()
        self._encoded_media_data = dict()
        self._encoder = None
        self._media_list = None

        self.__load_encoder()

    def __load_encoder(self):
        if self._baseline_args.baseline_encode_method == 'tfidf':
            n_gram = (self._baseline_args.min_num_gram, self._baseline_args.max_num_gram)
            self._encoder = TfidfVectorizer(ngram_range=n_gram)

    def load_data(self):
        data_path = self._data_args.data_dir
        media_list = os.listdir(data_path)
        self._media_list = media_list

        for media in self._media_list:
            if media not in self._original_media_data:
                self._original_media_data[media] = list()
            file_path = os.path.join(os.path.join(os.path.join(data_path, media),'original'),'en.valid')
            with open(file_path,mode='r',encoding='utf8') as fp:
                for line in fp.readlines():
                    paragraph = line.strip().replace('\\n\\n',' ')
                    sentence_list = sent_tokenize(paragraph)
                    self._original_media_data[media].extend(sentence_list)



    def encode_data(self):
        all_media_data = list()
        all_data = list()
        for media in self._media_list:
            all_media_data.append(' '.join(self._original_media_data[media]))
            all_data.extend(self._original_media_data[media])

        self._encoded_data = self._encoder.fit_transform(all_data)

    def feature_analysis(self) -> int:
        cluster = KMeans(n_clusters=len(self._media_list))
        ground_truth = list()
        ground_truth_adj = list()

        for media_name in self._media_list:
            label = self._media_list.index(media_name)
            ground_truth.extend([label for _ in range(len(self._original_media_data[media_name]))])
        predicted = cluster.fit_predict(self._encoded_data)
        result = adjusted_rand_score(ground_truth,predicted)
        print(result)
        return result

