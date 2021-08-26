import enum
import os
from numpy.core.fromnumeric import shape
from numpy.lib.utils import source
from scipy.spatial.kdtree import distance_matrix
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

from program.config import ArticleMap, TwitterMap, FullArticleMap, BaselineArticleMap


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
    data_map = BaselineArticleMap() if data_type=='article' else TwitterMap()

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
    plt.title('Baseline')
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




def main():
    for data_type in ['article']:
        source_model = build_baseline(data_type,'source')
        trust_model = build_baseline(data_type,'trust')
    temp(source_model, trust_model)
if __name__ == '__main__':
    main()