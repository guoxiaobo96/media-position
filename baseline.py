import os
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import csv
from copy import deepcopy
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics.pairwise import euclidean_distances
import joblib
from scipy.stats import kendalltau
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class BaselineArticleMap:
    dataset_to_name: Dict = field(default_factory=lambda: {'Breitbart': 'Breitbart', 'CBS': 'CBS News', 'CNN': 'CNN', 'Fox': 'Fox News', 'HuffPost': 'HuffPost',
                                  'NPR': 'NPR', 'NYtimes': 'New York Times', 'usatoday': 'USA Today', 'wallstreet': 'Wall Street Journal', 'washington': 'Washington Post'})
    name_to_dataset: Dict = field(init=False)
    dataset_list: List[str] = field(init=False)
    dataset_bias: Dict = field(default_factory=lambda: {'Breitbart': 2, 'CBS': -1, 'CNN': -5/3, 'Fox': 5/3,
                               'HuffPost': -2, 'NPR': -0.5, 'NYtimes': -1.5, 'usatoday': -1, 'wallstreet': 0.5, 'washington': -1})
    left_dataset_list: List[str] = field(
        default_factory=lambda: ['Breitbart', 'Fox', 'sean', 'rushlimbaugh.com'])

    def __post_init__(self):
        self.name_to_dataset = {v: k for k, v in self.dataset_to_name.items()}
        self.dataset_list = [k for k, v in self.dataset_to_name.items()]


def print_figure():
    label_list = ["Breitbart", "CBS", "CNN", "Fox", "Huffpost",
                  "NPR", "NYtimes", "usatoday", "wallstreet", "washington"]
    model_file = '/home/xiaobo/media-position/analysis/obamacare/obamacare/42/mlm/bigram_outer/sentence_order_replacement/4/cluster/model.c'
    model = joblib.load(model_file)
    plt.title('Ours')
    plot_dendrogram(model, orientation='right',
                    labels=label_list)
    plt_file = 'temp.png'
    plt.savefig(plt_file, bbox_inches='tight')
    plt.close()


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
    dendrogram(linkage_matrix, color_threshold=0, **kwargs)


def cluster_generate(model: AgglomerativeClustering, label_list=None):
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


def build_baseline(label_type):
    data_map = BaselineArticleMap()

    data_map = BaselineArticleMap()
    bias_distance_matrix = np.zeros(
        shape=(len(data_map.dataset_bias), len(data_map.dataset_bias)))
    distance_order_matrix = np.zeros(
        shape=(len(data_map.dataset_bias), len(data_map.dataset_bias)), dtype=int)
    for i, media_a in enumerate(data_map.dataset_list):
        temp_distance = list()
        for j, media_b in enumerate(data_map.dataset_list):
            bias_distance_matrix[i][j] = abs(
                data_map.dataset_bias[media_a] - data_map.dataset_bias[media_b])
            temp_distance.append(
                abs(data_map.dataset_bias[media_a] - data_map.dataset_bias[media_b]))
        distance_set = set(temp_distance)
        distance_set = sorted(list(distance_set))
        for o, d_o in enumerate(distance_set):
            for j, d_j in enumerate(temp_distance):
                if d_o == d_j:
                    distance_order_matrix[i][j] = o

    label_list = list(data_map.name_to_dataset.keys())
    data = list()
    data_temp = dict()
    with open('./data/ground-truth/'+label_type+'.csv', mode='r', encoding='utf8') as fp:
        reader = csv.reader(fp)
        header = next(reader)
        for row in reader:
            data_temp[row[0]] = [float(x.strip()) for x in row[1:]]
    for k, _ in data_map.name_to_dataset.items():
        try:
            data_item = deepcopy(data_temp[k])
            for i in range(1, len(data_item)):
                data_item[i] /= data_item[0]
            data.append(data_item)
        except:
            print(k)
    media_distance = np.zeros(
        shape=(len(data_map.dataset_list), len(data_map.dataset_list)))
    for i, data_i in enumerate(data):
        for j, data_j in enumerate(data):
            media_distance[i][j] = euclidean_distances(
                np.array(data_i).reshape(1, -1), np.array(data_j).reshape(1, -1))
    media_distance_order_matrix = np.zeros(
        shape=(len(data_map.dataset_bias), len(data_map.dataset_bias)), dtype=int)
    for i, media_a in enumerate(data_map.dataset_list):
        temp_distance = list()
        for j, media_b in enumerate(data_map.dataset_list):
            temp_distance.append(media_distance[i][j])
        order_list = np.argsort(temp_distance)
        order_list = order_list.tolist()
        for j in range(len(data_map.dataset_list)):
            order = order_list.index(j)
            media_distance_order_matrix[i][j] = order
    sort_distance = 0
    for i in range(len(data_map.dataset_list)):
        tau, p_value = kendalltau(media_distance_order_matrix[i].reshape(
            1, -1), distance_order_matrix[i].reshape(1, -1))
        sort_distance += tau
    sort_distance /= len(data_map.dataset_list)

    analyzer = AgglomerativeClustering(
        n_clusters=2, compute_distances=True, affinity='euclidean', linkage='complete')
    cluster_result = dict()
    clusters = analyzer.fit(data)
    labels = clusters.labels_
    for i, label in enumerate(labels.tolist()):
        if label not in cluster_result:
            cluster_result[label] = list()
        cluster_result[label].append(label_list[i])

    if not os.path.exists('./log/ground-truth/model/'):
        os.makedirs('./log/ground-truth/model/')
    model_file = './log/ground-truth/model/ground-truth_'+label_type+'.c'
    distance_file = './log/ground-truth/model/ground-truth_'+label_type+'.npy'
    np.save(distance_file, media_distance)
    joblib.dump(analyzer, model_file)
    label_list = ["Breitbart", "CBS", "CNN", "Fox", "Huffpost",
                  "NPR", "NYtimes", "usatoday", "wallstreet", "washington"]
    plot_dendrogram(analyzer, orientation='right',
                    labels=label_list)
    analysis_dir = './analysis/ground-truth/'
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    plt_file = analysis_dir+'/ground-truth_'+label_type+'.png'
    plt.savefig(plt_file, bbox_inches='tight')
    plt.close()
    return analyzer


def main():
    source_model = build_baseline('SoA-s')
    trust_model = build_baseline('SoA-t')


if __name__ == '__main__':
    main()
