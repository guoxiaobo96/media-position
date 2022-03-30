import os
from random import random
from matplotlib import pyplot as plt
import csv
from gensim.models import KeyedVectors
from nltk.stem.porter import *
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import Dict, Set, Tuple, List
import joblib
from grakel import Graph
from grakel.graph_kernels import *
from copy import deepcopy
from zss import simple_distance, Node
from scipy.stats import kendalltau
import random

from sklearn.cluster import (
    KMeans,
    AffinityPropagation,
    MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    Birch)
from sklearn.metrics import adjusted_mutual_info_score
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics.pairwise import(
    cosine_distances,
)
from scipy.sparse.csgraph import shortest_path
import numpy as np
from numpy import ndarray


from .config import AnalysisArguments, MiscArgument, ModelArguments, DataArguments, TrainingArguments
from .model import BertSimpleModel


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
        self._misc_args = misc_args
        self._data_args = data_args
        self._model_args = model_args
        self._training_args = training_args

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
        sentence_number: str,
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
    def __init__(self, misc_args: MiscArgument, model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments, config: AnalysisArguments, pre_computer=False) -> None:
        super().__init__(misc_args, model_args, data_args, training_args, config)
        self._load_analysis_model(
            self._config.analysis_cluster_method, pre_computer)

    def _load_analysis_model(
        self,
        cluster_method: str,
        pre_computer
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
            # self._analyser =  AgglomerativeClustering(n_clusters=2, compute_distances=True)
            if not pre_computer:
                self._analyser = AgglomerativeClustering(
                    n_clusters=2, compute_distances=True, affinity='euclidean', linkage='complete')
            else:
                self._analyser = AgglomerativeClustering(
                    n_clusters=2, compute_distances=True, affinity='precomputed', linkage='complete')
        elif cluster_method == "DBSCAN":
            self._analyser = DBSCAN(eps=0.5, min_samples=2)
        elif cluster_method == "OPTICS":
            self._analyser = OPTICS()
        elif cluster_method == "Birch":
            self._analyser = Birch()

    def analyze(
        self,
        data,
        sentence_number: str,
        analysis_args: AnalysisArguments,
        keep_result=True,
        encode: bool = True,
        dataset_list: List = [],

    ) -> Dict[int, Set[str]]:
        cluster_result = dict()
        if encode:
            if 'vanilla' in data:
                data.pop('vanilla')
            dataset_list, encoded_list = self._encode_data(data)
        else:
            encoded_list = data

        clusters = deepcopy(self._analyser.fit(encoded_list))
        labels = clusters.labels_
        for i, label in enumerate(labels.tolist()):
            if label not in cluster_result:
                cluster_result[label] = list()
            cluster_result[label].append(dataset_list[i])
        if keep_result:
            plt.title('Hierarchical Clustering Dendrogram')
            plot_dendrogram(self._analyser, orientation='right',
                            labels=dataset_list)
            plt_file = os.path.join(analysis_args.analysis_result_dir, analysis_args.analysis_encode_method +
                                    '_'+analysis_args.analysis_cluster_method+'_'+sentence_number+'.png')
            data_file = os.path.join(analysis_args.analysis_result_dir, analysis_args.analysis_encode_method +
                                     '_'+sentence_number+'_distance.npy')
            model_path = analysis_args.analysis_result_dir
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_file = os.path.join(model_path, 'model.c')
            joblib.dump(self._analyser, model_file)
            plt.savefig(plt_file, bbox_inches='tight')
            plt.close()
            np.save(data_file, data)
        return clusters, cluster_result, dataset_list, encoded_list


def plot_dendrogram(model, **kwargs):
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
        sentence_number: str,
        analysis_args: AnalysisArguments,
        keep_result=True
    ) -> None:
        distance_result = dict()
        dataset_list, encoded_list = self._encode_data(data)

        base_vector = encoded_list[dataset_list.index('vanilla')]
        exclusive_dataset_list = []
        exclusive_vector_list = []
        for i, vector in enumerate(encoded_list):
            if dataset_list[i] != 'vanilla':
                exclusive_vector_list.append(vector)
                exclusive_dataset_list.append(dataset_list[i])
        distance_list = np.squeeze(self._analyser(exclusive_vector_list))
        # distance_list = np.squeeze(self._analyser(
        #     [base_vector], exclusive_vector_list))

        # for i, distance in enumerate(distance_list.tolist()):
        #     distance_result[exclusive_dataset_list[i]] = distance

        return exclusive_dataset_list, distance_list


class CorrelationAnalysis(BaseAnalysis):
    def __init__(self,  misc_args: MiscArgument, model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments, config: AnalysisArguments) -> None:
        super().__init__(misc_args, model_args, data_args, training_args, config)
        self._load_analysis_model(self._config.analysis_correlation_method)

    def _load_analysis_model(
        self,
        method: str
    ):
        if method == "tau":
            self._analyser = self._rank_distance
        elif method == 'pearson':
            self._analyser = cosine_distances

    def _rank_distance(self, data, data_map):
        distance_matrix = cosine_distances(data)
        media_distance_order_matrix = np.zeros(
            shape=(len(data_map.dataset_bias), len(data_map.dataset_bias)), dtype=np.int)
        for i, media_a in enumerate(data_map.dataset_list):
            temp_distance = list()
            distance_map = list()
            for j, media_b in enumerate(data_map.dataset_list):
                # temp_distance.append(distance_matrix[i][j])
                distance_map.append((j, distance_matrix[i][j]))
            random.shuffle(distance_map)
            for item in distance_map:
                temp_distance.append(item[1])
            if temp_distance.count(0) > 1:
                return None
            order_list = np.argsort(temp_distance)
            order_list = order_list.tolist()

            for order, v in enumerate(order_list):
                media_distance_order_matrix[i][distance_map[v][0]] = order

            # order_list = np.argsort(temp_distance)
            # order_list = order_list.tolist()
            # for j in range(len(data_map.dataset_list)):
            #     order = distance_map[order_list.index(j)][1]
            #     media_distance_order_matrix[i][j] = order
        return media_distance_order_matrix

    def analyze(
        self,
        data,
        sentence_number: str,
        analysis_args: AnalysisArguments,
        keep_result=True,
        encode: bool = True,
        dataset_list: List = [],
        data_map=None,

    ) -> Dict[int, Set[str]]:
        cluster_result = dict()
        if encode:
            if 'vanilla' in data:
                data.pop('vanilla')
            dataset_list, encoded_list = self._encode_data(data)
        else:
            encoded_list = data
        if self._config.analysis_correlation_method == 'tau':
            media_distance_matrix = self._analyser(encoded_list, data_map)
        elif self._config.analysis_correlation_method == 'pearson':
            media_distance_matrix = self._analyser(encoded_list)
        return media_distance_matrix, media_distance_matrix, dataset_list, encoded_list


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
        self._model = BertSimpleModel(model_args, data_args, training_args)

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


class CorrelationCompare(object):
    def __init__(self, misc_args: MiscArgument, analysis_args: AnalysisArguments) -> None:
        super().__init__()
        self. _result_path = os.path.join(os.path.join(
            analysis_args.analysis_result_dir, analysis_args.graph_distance), analysis_args.graph_kernel)
        self._analysis_args = analysis_args

    def compare(self, model_dict):
        name_list = list()
        model_list = list()
        result_dict = dict()

        for name, model in model_dict.items():
            name_list.append(name)
            model_list.append(model)

        base_index = name_list.index('base')
        step_size = 0.05
        distribution = [0 for _ in range(int(2/step_size) + 1)]
        for k, name in enumerate(tqdm(name_list, desc="Calculate distance")):
            if k == base_index:
                continue
            performance = 0
            if self._analysis_args.analysis_correlation_method == 'tau':
                for i in range(len(model_list[base_index])):
                    tau, p_value = kendalltau(model_list[k][i].reshape(
                        1, -1), model_list[base_index][i].reshape(1, -1))
                    performance += tau
            elif self._analysis_args.analysis_correlation_method == 'pearson':
                for i in range(len(model_list[base_index])):
                    pearson = np.corrcoef(model_list[k][i].reshape(
                        1, -1), model_list[base_index][i].reshape(1, -1))
                    performance += pearson[0][1]
            performance /= len(model_list[base_index])
            result_dict[name] = performance
        return result_dict


class ClusterCompare(object):
    def __init__(self, misc_args: MiscArgument, analysis_args: AnalysisArguments) -> None:
        super().__init__()
        self. _result_path = os.path.join(os.path.join(
            analysis_args.analysis_result_dir, analysis_args.graph_distance), analysis_args.graph_kernel)
        self._analysis_args = analysis_args

    def _calculate_leaf_distance(self, model: AgglomerativeClustering):
        leaf_node_number = len(model.labels_)
        inter_node_number = len(model.children_)
        counts = np.zeros(leaf_node_number+inter_node_number)
        for i in range(leaf_node_number):
            counts[i] = 1

        node_matrix = np.zeros(
            (leaf_node_number+inter_node_number, leaf_node_number+inter_node_number))
        distance_matrix = np.zeros((leaf_node_number, leaf_node_number))
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                for distance in self._analysis_args.graph_distance.split('_'):
                    if distance == 'cluster':
                        node_matrix[i+leaf_node_number][child_idx] += model.distances_[i]
                        node_matrix[child_idx][i +
                                               leaf_node_number] += model.distances_[i]
                    elif distance == 'alpha':
                        node_matrix[i+leaf_node_number][child_idx] += 1
                        node_matrix[child_idx][i+leaf_node_number] += 1
                    elif distance == 'acc':
                        node_matrix[i+leaf_node_number][child_idx] += counts[child_idx]
                        node_matrix[child_idx][i +
                                               leaf_node_number] += counts[child_idx]
                    current_count += counts[child_idx]
                counts[i+leaf_node_number] = current_count

        dist_matrix = shortest_path(csgraph=node_matrix, directed=False)
        dist_matrix = dist_matrix[:leaf_node_number, :leaf_node_number]

        return dist_matrix

    def _graph_generate(self, model: AgglomerativeClustering, label_list: List[int] = None):
        edges = list()
        edge_labels = dict()
        node_labels = dict()
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        if label_list is None:
            label_list = [i for i in range(n_samples)]

        for i, merge in enumerate(model.children_):
            current_counts = 0
            for child_idx in merge:
                distance = 0
                for distance_type in self._analysis_args.graph_distance.split('_'):
                    if distance_type == 'cluster':
                        distance += model.distances_[i]
                        distance += model.distances_[i]
                    elif distance_type == 'alpha':
                        distance += 1
                        distance += 1
                edges.append((i+n_samples, child_idx, distance))
                edge_labels[(i+n_samples, child_idx)] = 1
                if child_idx < n_samples:
                    current_counts += 1
                else:
                    current_counts += counts[child_idx - n_samples]
            counts[i] = current_counts
        for i in range(len(counts)+n_samples):
            if i < n_samples:
                node_labels[i] = label_list[i]
            else:
                node_labels[i] = n_samples
        graph = Graph(edges, node_labels=node_labels, edge_labels=edge_labels)
        return graph

    def _cluster_generate(self, model: AgglomerativeClustering, label_list: List[int] = None):
        cluster_dict = dict()
        n_samples = len(model.labels_)
        if label_list is None:
            label_list = [i for i in range(n_samples)]
        for i, merge in enumerate(model.children_):
            cluster_set = set()
            for child_idx in merge:
                if child_idx < n_samples:
                    cluster_set.add(label_list[child_idx])
                else:
                    cluster_set = cluster_set | cluster_dict[child_idx]
            cluster_dict[i+n_samples] = cluster_set
        cluster_list = list(cluster_dict.values())
        return cluster_list

    def _tree_generate(self, model: AgglomerativeClustering, label_list: List[int] = None):
        cluster_dict = dict()
        n_samples = len(model.labels_)
        if label_list is None:
            label_list = [i for i in range(n_samples)]

        node_list = [0 for _ in range(n_samples*2 - 1)]
        for i in range(n_samples):
            node_list[i] = Node(str(i), [])
        for i, merge in enumerate(model.children_):
            child_list = list()
            for child_idx in merge:
                child_list.append(node_list[child_idx])
            node_list[i+n_samples] = Node('-1', child_list)
        return node_list[-1]

    def _build_graph(self, model, label_list=None) -> None:
        if self._analysis_args.graph_kernel == 'cluster':
            if self._analysis_args.graph_distance in ['count', 'co_occurance']:
                return self._cluster_generate(model)
            elif self._analysis_args.graph_distance in ['adjusted_rand_index']:
                return model.labels_
            else:
                return self._calculate_leaf_distance(model)
        elif self._analysis_args.graph_kernel == 'tree':
            return self._tree_generate(model)
        else:
            return self._graph_generate(model, label_list)

    def _calculate_distance(self, graph_list, name_list) -> None:
        result_dict = dict()
        if self._analysis_args.graph_kernel not in ['cluster', 'tree']:
            if self._analysis_args.graph_kernel == 'WeisfeilerLehman':
                gk = WeisfeilerLehman()
            elif self._analysis_args.graph_kernel == 'GraphletSampling':
                gk = GraphletSampling()
            elif self._analysis_args.graph_kernel == 'RandomWalk':
                gk = RandomWalk()
            elif self._analysis_args.graph_kernel == 'RandomWalkLabeled':
                gk = RandomWalkLabeled()
            elif self._analysis_args.graph_kernel == 'ShortestPath':
                gk = ShortestPath()
            elif self._analysis_args.graph_kernel == 'ShortestPathAttr':
                gk = ShortestPathAttr()
            elif self._analysis_args.graph_kernel == 'NeighborhoodHash':
                gk = NeighborhoodHash()
            elif self._analysis_args.graph_kernel == 'PyramidMatch':
                gk = PyramidMatch()
            elif self._analysis_args.graph_kernel == 'SubgraphMatching':
                gk = SubgraphMatching()
            elif self._analysis_args.graph_kernel == 'NeighborhoodSubgraphPairwiseDistance':
                gk = NeighborhoodSubgraphPairwiseDistance()
            elif self._analysis_args.graph_kernel == 'LovaszTheta':
                gk = LovaszTheta()
            elif self._analysis_args.graph_kernel == 'SvmTheta':
                gk = SvmTheta()
            elif self._analysis_args.graph_kernel == 'OddSth':
                gk = OddSth()
            elif self._analysis_args.graph_kernel == 'Propagation':
                gk = Propagation()
            elif self._analysis_args.graph_kernel == 'PropagationAttr':
                gk = PropagationAttr()
            elif self._analysis_args.graph_kernel == 'HadamardCode':
                gk = HadamardCode()
            elif self._analysis_args.graph_kernel == 'MultiscaleLaplacian':
                gk = MultiscaleLaplacian()
            elif self._analysis_args.graph_kernel == 'VertexHistogram':
                gk = VertexHistogram()
            elif self._analysis_args.graph_kernel == 'EdgeHistogram':
                gk = EdgeHistogram()
            elif self._analysis_args.graph_kernel == 'GraphHopper':
                gk = GraphHopper()
            elif self._analysis_args.graph_kernel == 'CoreFramework':
                gk = CoreFramework()
            elif self._analysis_args.graph_kernel == 'WeisfeilerLehmanOptimalAssignment':
                gk = WeisfeilerLehmanOptimalAssignment()

            graph_list = gk.fit_transform(graph_list)
        base_index = name_list.index('base')
        for i, name in enumerate(tqdm(name_list, desc="Calculate distance")):
            if i != base_index:
                if self._analysis_args.graph_distance == 'count':
                    count = 0
                    for cluster in graph_list[i]:
                        if cluster not in graph_list[base_index]:
                            count += 1
                    distance = count / (len(graph_list[base_index]) - 1)
                elif self._analysis_args.graph_distance == 'co_occurance':
                    distance = self._co_occurance_distance(
                        graph_list[i], graph_list[base_index])
                elif self._analysis_args.graph_distance == 'adjusted_rand_index':
                    distance = adjusted_mutual_info_score(
                        graph_list[base_index], graph_list[i])
                elif self._analysis_args.graph_distance == 'edit':
                    distance = simple_distance(
                        graph_list[base_index], graph_list[i])
                else:
                    distance = np.linalg.norm(
                        graph_list[i]-graph_list[base_index])

                result_dict[name] = distance
        return result_dict

    def _co_occurance_distance(self, cluster_list, base_cluster_list):
        cluster_number = len(cluster_list) - 1
        distance_matrix = np.zeros((len(cluster_list)+1, len(cluster_list)+1))
        basic_distance_matrix = np.zeros(
            (len(cluster_list)+1, len(cluster_list)+1))

        for cluster in cluster_list:
            if len(cluster) == len(cluster_list) + 1:
                continue
            for i in cluster:
                for j in cluster:
                    if i != j:
                        distance_matrix[i][j] += 1
        # for i in range(len(cluster_list)+1):
        #     temp = np.sum(distance_matrix[i],axis=0)
        #     if temp != 0:
        #         distance_matrix[i] = distance_matrix[i] / temp

        for cluster in base_cluster_list:
            if len(cluster) == len(cluster_list) + 1:
                continue
            for i in cluster:
                for j in cluster:
                    if i != j:
                        basic_distance_matrix[i][j] += 1
        # for i in range(len(cluster_list)+1):
        #     basic_distance_matrix[i] = basic_distance_matrix[i] / np.sum(basic_distance_matrix[i],axis=0)
        distance = 0
        for i in range(len(distance_matrix)):
            distance += cosine_distances(distance_matrix[i].reshape(
                1, -1), basic_distance_matrix[i].reshape(1, -1))
            # distance += manhattan_distances(distance_matrix[i].reshape(1,-1), basic_distance_matrix[i].reshape(1,-1))
        return distance[0][0]

    def compare(self, model_dict: Dict[str, AgglomerativeClustering], label_list=None) -> None:
        graph_list = list()
        name_list = list()
        method = str()
        if self._analysis_args.analysis_compare_method == 'cluster':
            method = self._analysis_args.analysis_cluster_method
        elif self._analysis_args.analysis_compare_method == 'distance':
            method = self._analysis_args.analysis_distance_method

        for name, model in model_dict.items():
            name_list.append(name)
            graph_list.append(self._build_graph(model, label_list))

        result_dict = self._calculate_distance(graph_list, name_list)

        return result_dict


def main():
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
