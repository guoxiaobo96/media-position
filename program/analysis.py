import logging
import warnings
import math
import os
from matplotlib import pyplot as plt
import csv
from dataclasses import dataclass, field
from gensim.models import Word2Vec, KeyedVectors
from nltk.stem.porter import *
from abc import ABC, abstractmethod
from glob import glob
from typing import Any, Dict, Optional, Set, Tuple, Union, List
from sklearn import cluster

from sklearn.cluster import (
    KMeans,
    AffinityPropagation,
    MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    Birch)
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics.pairwise import(
    cosine_distances
)


import numpy as np
from numpy import ndarray


from tokenizers import Tokenizer, models


from .config import AnalysisArguments, MiscArgument, ModelArguments, DataArguments, TrainingArguments
from .model import BertModel, encode_bert


class BaseAnalysis(ABC):
    def __init__(
        self,
        misc_args: MiscArgument,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        config: AnalysisArguments

    ) -> None:
        self._config = config
        self._encoder = None
        self._analyser = None

        self._load_encoder(self._config.analysis_encode_method, misc_args,
                           model_args, data_args, training_args)

    def _load_encoder(
        self,
        encode_method: str,
        misc_args: MiscArgument,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments
    ) -> None:
        if encode_method == "term":
            self._encoder = TermEncoder()
        elif encode_method == "bert":
            self._encoder = BertEncoder(model_args, data_args, training_args)
        elif encode_method == "word2vec":
            self._encoder = Word2VecEncoder()
        elif encode_method == 'liwc':
            self._encoder = LiwcEncoder(misc_args, data_args)
        elif encode_method == "binary":
            self._encoder = BinaryEncoder()

    @abstractmethod
    def _load_analysis_model(
        self,
        compare_method: str
    ):
        pass

    @abstractmethod
    def analyze(
        self,
        data,
        sentence_number : str,
        analysis_args: AnalysisArguments
    ):
        pass

    def _encode_data(
        self,
        data
    ) -> Tuple[List[str], List[ndarray]]:
        encoded_result = self._encoder.encode(data)
        dataset_list = list(encoded_result.keys())
        encoded_list = list(encoded_result.values())
        return dataset_list, encoded_list


class ClusterAnalysis(BaseAnalysis):
    def __init__(self, misc_args: MiscArgument, model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments, config: AnalysisArguments) -> None:
        super().__init__(misc_args, model_args, data_args, training_args, config)
        self._load_analysis_model(self._config.analysis_cluster_method)

    def _load_analysis_model(
        self,
        cluster_method: str
    ):
        if cluster_method == "KMeans":
            self._analyser = KMeans()
        elif cluster_method == "AffinityPropagation":
            self._analyser = AffinityPropagation()
        elif cluster_method == "MeanShift":
            self._analyser = MeanShift()
        elif cluster_method == SpectralClustering():
            self._analyser = SpectralClustering()
        elif cluster_method == "AgglomerativeClustering":
            self._analyser = AgglomerativeClustering(n_clusters=2, compute_distances=True)
        elif cluster_method == "DBSCAN":
            self._analyser = DBSCAN(eps=0.5, min_samples=2)
        elif cluster_method == "OPTICS":
            self._analyser = OPTICS()
        elif cluster_method == "Birch":
            self._analyser = Birch()

    def analyze(
        self,
        data,
        sentence_number : str,
        analysis_args: AnalysisArguments
    ) -> Dict[int, Set[str]]:
        cluster_result = dict()
        dataset_list, encoded_list = self._encode_data(data)
        clusters = self._analyser.fit(encoded_list)
        labels = clusters.labels_
        for i, label in enumerate(labels.tolist()):
            if label not in cluster_result:
                cluster_result[label] = list()
            cluster_result[label].append(dataset_list[i])
        plt.title('Hierarchical Clustering Dendrogram')
        plot_dendrogram(self._analyser, orientation='right', labels=dataset_list)
        plt_file = os.path.join(analysis_args.analysis_result_dir,analysis_args.analysis_encode_method+'_'+analysis_args.analysis_cluster_method+'_'+sentence_number+'.png')
        plt.savefig(plt_file,bbox_inches = 'tight')
        plt.close()
        return cluster_result


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


class DistanceAnalysis(BaseAnalysis):
    def __init__(self,  misc_args: MiscArgument, model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments, config: AnalysisArguments) -> None:
        super().__init__(misc_args, model_args, data_args, training_args, config)
        self._load_analysis_model(self._config.analysis_distance_method)

    def _load_analysis_model(
        self,
        distance_method: str
    ):
        if distance_method == "Cosine":
            self._analyser = cosine_distances

    def analyze(
        self,
        data,
        sentence_number : str,
        analysis_args: AnalysisArguments
    ) -> Dict[int, Set[str]]:
        distance_result = dict()
        dataset_list, encoded_list = self._encode_data(data)

        base_vector = encoded_list[dataset_list.index('vanilla')]
        exclusive_dataset_list = []
        exclusive_vector_list = []
        for i, vector in enumerate(encoded_list):
            if dataset_list[i] != 'vanilla':
                exclusive_vector_list.append(vector)
                exclusive_dataset_list.append(dataset_list[i])
        distance_list = np.squeeze(self._analyser(
            [base_vector], exclusive_vector_list))

        for i, distance in enumerate(distance_list.tolist()):
            distance_result[exclusive_dataset_list[i]] = distance

        return distance_result


class TermEncoder(object):
    def __init__(self) -> None:
        self._term_dict = dict()

    def encode(
        self,
        data: Dict
    ) -> Dict[str, Dict]:
        term_set = set()
        encode_result = dict()
        for _, term_dict in data.items():
            term_set = term_set.union(set(term_dict.keys()))
        for i, term in enumerate(list(term_set)):
            self._term_dict[term] = i
        for dataset, term_dict in data.items():
            encode_array = np.zeros(shape=len(term_set))
            for k, v in term_dict.items():
                encode_array[self._term_dict[k]] = float(v)
            encode_result[dataset] = encode_array
        return encode_result

class BinaryEncoder(object):
    def __init__(self) -> None:
        self._term_dict = dict()

    def encode(
        self,
        data: Dict
    ) -> Dict[str, Dict]:
        term_set = set()
        encode_result = dict()
        for _, term_dict in data.items():
            term_set = term_set.union(set(term_dict.keys()))
        for i, term in enumerate(list(term_set)):
            self._term_dict[term] = i
        for dataset, term_dict in data.items():
            encode_array = np.zeros(shape=len(term_set))
            for k, v in term_dict.items():
                encode_array[self._term_dict[k]] = 1
            encode_result[dataset] = encode_array
        return encode_result


class LiwcEncoder(object):
    def __init__(
        self,
        misc_args: MiscArgument,
        data_args: DataArguments
    ) -> None:
        self._term_dict = dict()
        self._log_dir = os.path.join(misc_args.log_dir, data_args.data_type)
        self.load_dict()

    def load_dict(
        self,
    ) -> None:
        category_file = os.path.join(os.path.join(
            self._log_dir, 'dict'), 'category.csv')
        with open(category_file, mode='r') as fp:
            reader = csv.reader(fp)
            for row in reader:
                if 'Word' not in row:
                    category = [0 for _ in range(len(row)-1)]
                    for i, mark in enumerate(row):
                        if mark == 'X':
                            category[i-1] = 1
                    self._term_dict[row[0]] = category

    def encode(
        self,
        data: Dict
    ) -> Dict[str, Dict]:
        encode_result = dict()
        for dataset, term_dict in data.items():
            term_list = list(term_dict.keys())
            score_list = np.array(list(term_dict.values()), dtype=np.float)
            score_list = score_list / np.sum(score_list)
            term_encode = [self._term_dict[term.lower()] for term in term_list]
            term_encode = np.array(term_encode)
            term_encode = term_encode.T*score_list
            encode_result[dataset] = np.sum(term_encode.T, axis=0)
        return encode_result


class BertEncoder(object):
    def __init__(self, model_args, data_args, training_args) -> None:
        self._model = BertModel(model_args, data_args, training_args)

    def encode(
        self,
        data: Dict
    ) -> Dict[str, Dict]:
        encode_result = dict()
        for dataset, term_dict in data.items():
            term_list = list(term_dict.keys())
            score_list = np.array(list(term_dict.values()), dtype=np.float)
            score_list = score_list / np.sum(score_list)
            term_encode = self._model.encode(term_list)
            term_encode = np.squeeze(np.array(list(term_encode.values())))
            term_encode = term_encode.T*score_list
            encode_result[dataset] = np.sum(term_encode.T, axis=0)
        return encode_result


class Word2VecEncoder(object):
    def __init__(self) -> None:
        self._model = KeyedVectors.load_word2vec_format(
            "/home/xiaobo/pretrained_models/word2vec.bin", binary=True)

    def encode(
        self,
        data: Dict
    ) -> Dict[str, Dict]:
        term_set = set()
        encode_result = dict()
        stemmer = PorterStemmer()
        for dataset, term_dict in data.items():
            term_encode = []
            score_list = []
            for term, score in term_dict.items():
                if stemmer.stem(term) in self._model.vocab:
                    term_encode.append(self._model[stemmer.stem(term)])
                    score_list.append(score)

            score_list = np.array(score_list, dtype=np.float)
            score_list = score_list / np.sum(score_list)
            term_encode = np.array(term_encode, dtype=np.float)
            # term_encode = np.squeeze(np.array(list(term_encode.values())))
            term_encode = term_encode.T*score_list
            encode_result[dataset] = np.sum(term_encode.T, axis=0)
        return encode_result




def main():
    # from config import get_config
    # from data import get_analysis_data
    # from util import prepare_dirs_and_logger
    # misc_args, model_args, data_args, training_args, adapter_args, analysis_args = get_config()
    # prepare_dirs_and_logger(misc_args, model_args,
    #                         data_args, training_args, adapter_args,analysis_args)
    # analysis_data = get_analysis_data(analysis_args)
    # analysis_model = DistanceAnalysis(model_args, data_args, training_args, analysis_args)
    # for k,v in analysis_data.items():
    #     analysis_model.analyze(analysis_data['4.json'])
    log_dir = '../../log/tweets'
    category_file = os.path.join(os.path.join(log_dir, 'dict'), 'category.csv')
    with open(category_file, mode='r') as fp:
        reader = csv.reader(fp)
        for row in reader:
            if 'Word' in row:
                continue
            else:
                category = [0 for _ in range(len(row)-1)]
                for i, mark in enumerate(row):
                    if mark == 'X':
                        category[i-1] = 1


if __name__ == '__main__':
    main()
