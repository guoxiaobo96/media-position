import os
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import csv
from copy import deepcopy
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns
import pandas as pd
import joblib
from scipy.stats import kendalltau
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, NewType


@dataclass
class BaselineArticleMap:
    dataset_to_name: Dict = field(default_factory=lambda: {'Breitbart':'Breitbart','CBS':'CBS News','CNN':'CNN','Fox':'Fox News','HuffPost':'HuffPost','NPR':'NPR','NYtimes':'New York Times','usatoday':'USA Today','wallstreet':'Wall Street Journal','washington':'Washington Post'})
    name_to_dataset: Dict = field(init=False)
    dataset_list: List[str] = field(init=False)
    dataset_bias: Dict = field(default_factory=lambda:{'Breitbart':2,'CBS':-1,'CNN':-5/3,'Fox':5/3,'HuffPost':-2,'NPR':-0.5,'NYtimes':-1.5,'usatoday':-1,'wallstreet':0.5,'washington':-1})
    left_dataset_list: List[str] = field(default_factory=lambda:['Breitbart', 'Fox', 'sean','rushlimbaugh.com'])

    def __post_init__(self):
        self.name_to_dataset = {v: k for k, v in self.dataset_to_name.items()}
        self.dataset_list = [k for k,v in self.dataset_to_name.items()]

def print_figure():
    label_list = ["Breitbart", "CBS","CNN","Fox","Huffpost","NPR","NYtimes","usatoday","wallstreet","washington"]
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


def cluster_generate(model: AgglomerativeClustering, label_list = None):
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


def distance_calculate(cluster_list):
    cluster_number = len(cluster_list) - 1
    distance_matrix = np.zeros((len(cluster_list)+1,len(cluster_list)+1))

    for cluster in cluster_list:
        if len(cluster) ==  len(cluster_list) + 1:
            continue
        for i in cluster:
            for j in cluster:
                if i != j:
                    distance_matrix[i][j] += 1
    return distance_matrix

def temp(source_model, trust_model):
    # source_model = '/home/xiaobo/media-position/log/baseline/model/baseline_source_article.c'
    # trust_model = '/home/xiaobo/media-position/log/baseline/model/baseline_trust_article.c'

    # source_model = joblib.load(source_model)
    # trust_model = joblib.load(trust_model)
    source_cluster_list = cluster_generate(source_model)
    trust_cluser_list = cluster_generate(trust_model)
    source_distance = distance_calculate(source_cluster_list)
    trust_distance = distance_calculate(trust_cluser_list)
    distance = 0
    for i in range(len(source_distance)):
        distance += cosine_distances(source_distance[i].reshape(1,-1), trust_distance[i].reshape(1,-1))
    print(distance[0][0])

def build_baseline(data_type, label_type):
    # data_map = BaselineArticleMap() if data_type=='article' else TwitterMap()
    data_map = BaselineArticleMap()

    data_map = BaselineArticleMap()
    bias_distance_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)))
    distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int)
    for i,media_a in enumerate(data_map.dataset_list):
        temp_distance = list()
        for j,media_b in enumerate(data_map.dataset_list):
            bias_distance_matrix[i][j] = abs(data_map.dataset_bias[media_a] - data_map.dataset_bias[media_b])
            temp_distance.append(abs(data_map.dataset_bias[media_a] - data_map.dataset_bias[media_b]))
        distance_set = set(temp_distance)
        distance_set = sorted(list(distance_set))
        for o, d_o in enumerate(distance_set):
            for j,d_j in enumerate(temp_distance):
                if d_o == d_j:
                    distance_order_matrix[i][j] = o



    label_list = list(data_map.name_to_dataset.keys())
    from sklearn.metrics import silhouette_score, silhouette_samples
    data = list()
    data_temp = dict()
    with open('./analysis/baseline/baseline_'+label_type+'_'+data_type+'.csv', mode='r', encoding='utf8') as fp:
        reader = csv.reader(fp)
        header = next(reader)
        for row in reader:
            data_temp[row[0]] = [float(x.strip()) for x in row[1:]]
    for k, _ in data_map.name_to_dataset.items():
        try:
            data_item = deepcopy(data_temp[k])
            for i in range(1,len(data_item)):
                data_item[i] /= data_item[0]
            data.append(data_item)
        except:
            print(k)
    media_distance = np.zeros(shape=(len(data_map.dataset_list),len(data_map.dataset_list)))
    for i,data_i in enumerate(data):
        for j, data_j in enumerate(data):
            media_distance[i][j] = euclidean_distances(np.array(data_i).reshape(1,-1),np.array(data_j).reshape(1,-1))
    media_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int)
    for i,media_a in enumerate(data_map.dataset_list):
        temp_distance = list()
        for j,media_b in enumerate(data_map.dataset_list):
            temp_distance.append(media_distance[i][j])
        order_list = np.argsort(temp_distance)
        order_list = order_list.tolist()
        for j in range(len(data_map.dataset_list)):
            order = order_list.index(j)
            media_distance_order_matrix[i][j] = order
    sort_distance = 0
    for i in range(len(data_map.dataset_list)):
        tau, p_value = kendalltau(media_distance_order_matrix[i].reshape(1,-1), distance_order_matrix[i].reshape(1,-1))
        sort_distance += tau
    sort_distance /= len(data_map.dataset_list)

    analyzer = AgglomerativeClustering(
        n_clusters=2, compute_distances=True, affinity='euclidean', linkage='complete')
    # analyzer = KMeans(n_clusters=3)
    cluster_result = dict()
    clusters = analyzer.fit(data)
    labels = clusters.labels_
    for i, label in enumerate(labels.tolist()):
        if label not in cluster_result:
            cluster_result[label] = list()
        cluster_result[label].append(label_list[i])
    score = davies_bouldin_score(data,labels)
    print(score)

    if not os.path.exists('./log/baseline/model/'):
        os.makedirs('./log/baseline/model/')
    model_file = './log/baseline/model/baseline_'+label_type+'_'+data_type+'.c'
    distance_file = './log/baseline/model/baseline_'+label_type+'_'+data_type+'.npy'
    np.save(distance_file,media_distance)
    joblib.dump(analyzer, model_file)
    if label_type == 'source':
        plt.title('SoA-s')
    else:
        plt.title('SoA-t')
    label_list = ["Breitbart", "CBS","CNN","Fox","Huffpost","NPR","NYtimes","usatoday","wallstreet","washington"]
    plot_dendrogram(analyzer, orientation='right',
                    labels=label_list)
    plt_file = './analysis/baseline/baseline_'+label_type+'_'+data_type+'.png'
    plt.savefig(plt_file, bbox_inches='tight')
    plt.close()

    # data = cosine_similarity(data)
    # data = pd.DataFrame(data,columns=label_list,index=label_list)
    # sns.heatmap(data)
    # plt_file = './analysis/baseline/baseline_'+data_type+'_heat.png'
    # plt.savefig(plt_file, bbox_inches='tight')
    # plt.close()
    return analyzer


def compare_human(ground_truth):
    human_media_list = [0,1,2,3,8]
    chosen_media_order = [[0,3,4,1,2],[4,0,1,3,2],[4,1,0,3,2],[1,3,4,0,2],[4,2,3,1,0]]
    data_map = BaselineArticleMap()
    bias_distance_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)))
    if ground_truth == "mbr":
        ground_truth_distance_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.float32)
        ground_truth_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int32)
        for i,media_a in enumerate(data_map.dataset_list):
            temp_distance = list()
            for j,media_b in enumerate(data_map.dataset_list):
                bias_distance_matrix[i][j] = abs(data_map.dataset_bias[media_a] - data_map.dataset_bias[media_b])
                temp_distance.append(abs(data_map.dataset_bias[media_a] - data_map.dataset_bias[media_b]))
                ground_truth_distance_matrix[i][j] = abs(data_map.dataset_bias[media_a] - data_map.dataset_bias[media_b])
            distance_set = set(temp_distance)
            distance_set = sorted(list(distance_set))
            for o, d_o in enumerate(distance_set):
                for j,d_j in enumerate(temp_distance):
                    if d_o == d_j:
                        ground_truth_distance_order_matrix[i][j] = o
    elif ground_truth in ['source','trust']:
        ground_truth_distance_matrix = np.load('E:/media-position/log/baseline/model/baseline_'+ground_truth+'_article.npy')
        ground_truth_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int32)
        for i,media_a in enumerate(data_map.dataset_list):
            temp_distance = ground_truth_distance_matrix[i]
            distance_set = set(temp_distance)
            distance_set = sorted(list(distance_set))
            for o, d_o in enumerate(distance_set):
                for j,d_j in enumerate(temp_distance):
                    if d_o == d_j:
                        ground_truth_distance_order_matrix[i][j] = o
    human_distance_order_matrix = np.array(chosen_media_order)
    human_distance_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int32)


    chosen_media_distance_order_matrix = np.zeros(shape=(5,5),dtype=np.int)
    for i, media_index in enumerate(human_media_list):
        chosen_media_distance_order_matrix[i] = ground_truth_distance_order_matrix[media_index,human_media_list]
    ground_truth_distance_order_matrix = chosen_media_distance_order_matrix
    media_count = 5

    tau_performance = 0
    for i in range(media_count):
        tau, p_value = kendalltau(human_distance_order_matrix[i].reshape(1,-1), ground_truth_distance_order_matrix[i].reshape(1,-1))
        tau_performance += tau
    tau_performance /= media_count

    print("The performance of {} is {}".format(ground_truth,tau_performance))

def main():
    # print_figure()
    # for data_type in ['article']:
    #     source_model = build_baseline(data_type,'source')
    #     trust_model = build_baseline(data_type,'trust')
    for ground_truth in ['source','trust','mbr']:
        compare_human(ground_truth)
    # temp(source_model, trust_model)
if __name__ == '__main__':
    main()