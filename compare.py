from json.decoder import JSONDecodeError
import os
from posixpath import join
import json
from tkinter import E
from typing_extensions import final
from gensim import corpora, models
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel, LdaMulticore
from scipy.stats import entropy
import numpy as np
from scipy.stats.stats import mode
from sklearn import cluster
import tqdm
import joblib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, NewType
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer


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

outlets_list = ['Breitbart', 'CBS', 'CNN', 'Fox', 'HuffPost', 'NPR', 'NYtimes', 'usatoday', 'wallstreet', 'washington']
data_type_list = ['train', 'eval', 'climate', 'obamacare']
aug_method_list = ["no_augmentation", "duplicate","sentence_order_replacement","sentence_replacement","word_order_replacement","word_replacement","span_cutoff"]


def _cluster_generate(model: AgglomerativeClustering, label_list: List[int] = None):
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

def _co_occurance_distance(cluster_list, base_cluster_list):
    cluster_number = len(cluster_list) - 1
    distance_matrix = np.zeros((len(cluster_list)+1,len(cluster_list)+1))
    basic_distance_matrix = np.zeros((len(cluster_list)+1,len(cluster_list)+1))

    for cluster in cluster_list:
        if len(cluster) ==  len(cluster_list) + 1:
            continue
        for i in cluster:
            for j in cluster:
                if i != j:
                    distance_matrix[i][j] += 1

    for cluster in base_cluster_list:
        if len(cluster) ==  len(cluster_list) + 1:
            continue
        for i in cluster:
            for j in cluster:
                if i != j:
                    basic_distance_matrix[i][j] += 1

    distance = 0
    for i in range(len(distance_matrix)):
        distance += cosine_distances(distance_matrix[i].reshape(1,-1), basic_distance_matrix[i].reshape(1,-1))

    return distance[0][0]

def lda_baseline(mean_method, file_list):
    def bd(vector_a, vector_b):
        bc = np.sum(
            np.sqrt(vector_a * vector_b))
        distance = -np.log(bc)
        return distance
    data_path = "e:/media-position/"
    topic_list = ["climate-change", "corporate-tax","drug-policy","gay-marriage","obamacare"]
    # topic_list = ["obamacare"]
    distance_dict = dict()
    for topic in topic_list:
        if mean_method == 'average':
            common_text = list()
            outlets_text_list = [[] for _ in range(10)]
            topic_path = os.path.join(data_path,'data_'+topic)
            topic_path = topic_path + '/42/all/original'
            for file_path in file_list:
                file = topic_path+'/'+file_path
                with open(file,mode='r',encoding='utf8') as fp:
                    for line in fp.readlines():
                        item = json.loads(line.strip())
                        text = item["sentence"].split(' ')
                        label = item["label"]
                        outlets_text_list[int(label)].append(text)
                        common_text.append(text)
                    common_dictionary = Dictionary(common_text)

            common_corpus = [common_dictionary.doc2bow(text) for text in common_text]
            outlets_text_corpus = list()
            for outlets_text in outlets_text_list:
                outlets_text_corpus.append([common_dictionary.doc2bow(text) for text in outlets_text])
            print("LDA running")
            n_topic = 10
            lda = LdaMulticore(common_corpus, num_topics=n_topic,random_state=42,workers=4,passes=2)
            print("LDA finish")
            outlets_vec_list = list()
            for outlets in tqdm.tqdm(outlets_text_corpus):
                t_list = list()
                for t in outlets:
                    outlets_vec_temp = lda[t]
                    outlets_vec = list(0 for _ in range(n_topic))
                    for item in outlets_vec_temp:
                        outlets_vec[item[0]] = item[1]
                    for i,_ in enumerate(outlets_vec):
                        if outlets_vec[i] == 0:
                            outlets_vec[i] = 1e-10
                    t_list.append(np.array(outlets_vec))
                t_distance  = np.mean(np.array(t_list),axis=0)
                outlets_vec_list.append(t_distance)
        elif mean_method == 'combine':
            common_text = list()
            outlets_text_list = [[] for _ in range(10)]
            topic_path = os.path.join(data_path,'data_'+topic)
            topic_path = topic_path + '/42/all/original'
            for file_path in file_list:
                file = topic_path+'/'+file_path
                with open(file,mode='r',encoding='utf8') as fp:
                    for line in fp.readlines():
                        item = json.loads(line.strip())
                        text = item["sentence"].split(' ')
                        label = item["label"]
                        outlets_text_list[int(label)].extend(text)
            for text in outlets_text_list:
                common_text.append(text)
            common_dictionary = Dictionary(common_text)

            common_corpus = [common_dictionary.doc2bow(text) for text in common_text]

            print("LDA running")
            n_topic = 10
            lda = LdaMulticore(common_corpus, num_topics=n_topic,random_state=42,workers=4,passes=2)
            print("LDA finish")
            outlets_vec_list = list()
            for outlets in common_corpus:
                outlets_vec_temp = lda[outlets]
                outlets_vec = list(0 for _ in range(n_topic))
                for item in outlets_vec_temp:
                    outlets_vec[item[0]] = item[1]
                for i,_ in enumerate(outlets_vec):
                    if outlets_vec[i] == 0:
                        outlets_vec[i] = 1e-10
                outlets_vec_list.append(np.array(outlets_vec))       

        distance_matrix = []
        for i, outlets_a_vec in enumerate(outlets_vec_list):
            d_list = [0 for _ in range(len(outlets_vec_list))]
            for j, outlets_b_vec  in enumerate(outlets_vec_list):
                if i!=j:
                    # distance = entropy(topic_b_vec,topic_a_vec)
                    # distance = bd(outlets_b_vec,outlets_a_vec)
                    distance = cosine_distances(outlets_b_vec.reshape(1, -1),outlets_a_vec.reshape(1, -1))[0][0]
                    d_list[j] = distance
            distance_matrix.append(np.array(d_list))
        distance_matrix = np.array(distance_matrix)
        distance_dict[topic] = distance_matrix
    return distance_dict

def tfidf_baseline(mean_method, file_list):
    data_path = "e:/media-position/"
    topic_list = ["obamacare","gay-marriage","drug-policy","corporate-tax","climate-change"]
    # topic_list = ["obamacare"]
    distance_dict = dict()
    for topic in topic_list:
        if mean_method == 'average':
            common_text = list()
            outlets_text_list = [[] for _ in range(10)]
            topic_path = os.path.join(data_path,'data_'+topic)
            topic_path = topic_path + '/42/all/original'
            for file_path in file_list:
                file = topic_path+'/'+file_path
                with open(file,mode='r',encoding='utf8') as fp:
                    for line in fp.readlines():
                        item = json.loads(line.strip())
                        text = item["sentence"].split(' ')
                        label = item["label"]
                        outlets_text_list[int(label)].append(' '.join(text))
                        common_text.append(' '.join(text))
            vectorizer = TfidfVectorizer(ngram_range=(1,3))
            model = vectorizer.fit(common_text)


            outlets_vec_list = list()
            for outlets in outlets_text_list:
                vec = model.transform(outlets)
                vec = np.mean(vec,axis=0)
                outlets_vec_list.append(vec)

        elif mean_method == 'combine':
            common_text = list()
            outlets_text_list = [[] for _ in range(10)]
            topic_path = os.path.join(data_path,'data_'+topic)
            topic_path = topic_path + '/42/all/original'
            for file_path in file_list:
                file = topic_path+'/'+file_path
                with open(file,mode='r',encoding='utf8') as fp:
                    for line in fp.readlines():
                        item = json.loads(line.strip())
                        text = item["sentence"].split(' ')
                        label = item["label"]
                        outlets_text_list[int(label)].extend(text)
            for text in outlets_text_list:
                common_text.append(' '.join(text))

            vectorizer = TfidfVectorizer()
            outlets_vec_list = vectorizer.fit_transform(common_text)
            outlets_vec_list = outlets_vec_list.todense()

        distance_matrix = []
        for i, outlets_a_vec in enumerate(outlets_vec_list):
            d_list = [0 for _ in range(len(outlets_vec_list))]
            for j, outlets_b_vec  in enumerate(outlets_vec_list):
                if i!=j:
                    # disance = entropy(topic_b_vec,topic_a_vec)
                    disance = cosine_distances(outlets_b_vec,outlets_a_vec)
                    d_list[j] = disance[0][0]
            distance_matrix.append(np.array(d_list))
        distance_matrix = np.array(distance_matrix)
        distance_dict[topic] = distance_matrix
    return distance_dict


def class_baseline():
    data_path = "/home/xiaobo/data/media-position/log/"
    topic_list = ["obamacare","gay-marriage","drug-policy","corporate-tax","climate-change"]
    # topic_list = ["obamacare"]
    distance_dict = dict()
    for topic in topic_list:
        outlets_vec_list = list()
        for outlet in outlets_list:
            file = os.path.join(os.path.join(os.path.join(data_path,topic),outlet),"class.npy")
            data = np.load(file)
            outlets_vec_list.append(data.reshape(1, -1))

        distance_matrix = []
        for i, outlets_a_vec in enumerate(outlets_vec_list):
            d_list = [0 for _ in range(len(outlets_vec_list))]
            for j, outlets_b_vec  in enumerate(outlets_vec_list):
                if i!=j:
                    # disance = entropy(topic_b_vec,topic_a_vec)
                    disance = cosine_distances(outlets_b_vec,outlets_a_vec)
                    d_list[j] = disance[0][0]
            distance_matrix.append(np.array(d_list))
        distance_matrix = np.array(distance_matrix)
        distance_dict[topic] = distance_matrix
    return distance_dict


def mlm_baseline():
    data_path = "/home/xiaobo/data/media-position/log/"
    # data_path = "/data/xiaobo/media-position/log/"
    topic_list = ["gay-marriage","drug-policy","corporate-tax","climate-change"]
    # topic_list = ["obamacare"]
    distance_dict = dict()
    for topic in topic_list:
        outlets_vec_list = list()
        for outlet in outlets_list:
            file = os.path.join(os.path.join(os.path.join(data_path,topic),outlet),"mlm.npy")
            data = np.load(file)
            outlets_vec_list.append(data.reshape(1, -1))

        distance_matrix = []
        for i, outlets_a_vec in enumerate(outlets_vec_list):
            d_list = [0 for _ in range(len(outlets_vec_list))]
            for j, outlets_b_vec  in enumerate(outlets_vec_list):
                if i!=j:
                    # disance = entropy(topic_b_vec,topic_a_vec)
                    disance = cosine_distances(outlets_b_vec,outlets_a_vec)
                    d_list[j] = disance[0][0]
            distance_matrix.append(np.array(d_list))
        distance_matrix = np.array(distance_matrix)
        distance_dict[topic] = distance_matrix
    return distance_dict

def get_baseline(ground_truth_list, file_list, method, combine_method):
    data_map = BaselineArticleMap()
    bias_distance_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)))
    if method == "tfidf":
        baseline_matrix_list = tfidf_baseline(combine_method,file_list)
    elif method == "lda":
        baseline_matrix_list = lda_baseline(combine_method,file_list)
    elif method == "mlm":
        baseline_matrix_list = mlm_baseline()

    for ground_truth in ground_truth_list:
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
        
        baseline_file = 'baseline_'+method+'_'+combine_method+'.json'
        performace_dict =  {'topic':'average','ground_truth':ground_truth,'tau_performance':[],'pearson_performance':[]}
        for topic, media_distance in baseline_matrix_list.items():
            # analyzer = AgglomerativeClustering(
            #     n_clusters=2, compute_distances=True, affinity='euclidean', linkage='complete')
            # # analyzer = KMeans(n_clusters=3)
            # cluster_result = dict()
            # clusters = analyzer.fit(media_distance)
            # labels = clusters.labels_
            # for i, label in enumerate(labels.tolist()):
            #     if label not in cluster_result:
            #         cluster_result[label] = list()
            #     cluster_result[label].append(data_map.dataset_list)
            # cluster_list = _cluster_generate(clusters)

            media_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int32)
            for i,media_a in enumerate(data_map.dataset_list):
                temp_distance = list()
                for j,media_b in enumerate(data_map.dataset_list):
                    temp_distance.append(media_distance[i][j])
                order_list = np.argsort(temp_distance)
                order_list = order_list.tolist()
                for j in range(len(data_map.dataset_list)):
                    order = order_list.index(j)
                    media_distance_order_matrix[i][j] = order
            
            
            
            
            tau_performance = 0
            for i in range(len(data_map.dataset_list)):
                tau, p_value = kendalltau(media_distance_order_matrix[i].reshape(1,-1), ground_truth_distance_order_matrix[i].reshape(1,-1))
                tau_performance += tau
            tau_performance /= len(data_map.dataset_list)

            pearson_performance = 0
            for i in range(len(data_map.dataset_list)):
                # sort_distance += euclidean_distances(media_distance_order_matrix[i].reshape(1,-1), distance_order_matrix[i].reshape(1,-1))
                pearson = np.corrcoef(ground_truth_distance_matrix[i].reshape(1,-1),media_distance[i].reshape(1,-1))
                pearson_performance += pearson[0][1]
            pearson_performance /= len(data_map.dataset_list)


            record_item = {'topic':topic,'ground_truth':ground_truth,'tau_performance':round(tau_performance,2),'pearson_performance':round(pearson_performance,2)}

            performace_dict['tau_performance'].append(round(tau_performance,2))
            performace_dict['pearson_performance'].append(round(pearson_performance,2))
            with open(baseline_file,mode='a',encoding='utf8') as fp:
                fp.write(json.dumps(record_item,ensure_ascii=False)+'\n')
        
        performace_dict['tau_performance'] = str(round(np.mean(performace_dict['tau_performance']),2)) + "("+str(round(np.std(performace_dict['tau_performance'],ddof=1),2))+")"
        performace_dict['pearson_performance'] = str(round(np.mean(performace_dict['pearson_performance']),2)) + "("+str(round(np.std(performace_dict['pearson_performance'],ddof=1),2))+")"
        with open(baseline_file,mode='a',encoding='utf8') as fp:
            fp.write(json.dumps(performace_dict,ensure_ascii=False)+'\n')

def baseline_difference():
    data_map = BaselineArticleMap()
    bias_distance_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)))
    allsides_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int32)
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
                    allsides_distance_order_matrix[i][j] = o

    trust_baseline_model = joblib.load(
        'E:/media-position/log/baseline/model/baseline_trust_article.c')
    source_baseline_model = joblib.load(
        'E:/media-position/log/baseline/model/baseline_source_article.c')

    trust_cluster_list = _cluster_generate(trust_baseline_model)       
    source_cluster_list = _cluster_generate(source_baseline_model)       

    trust_pew_distance_matrix = np.load('E:/media-position/log/baseline/model/baseline_trust_article.npy')
    trust_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int32)
    for i,media_a in enumerate(data_map.dataset_list):
        temp_distance = trust_pew_distance_matrix[i]
        distance_set = set(temp_distance)
        distance_set = sorted(list(distance_set))
        for o, d_o in enumerate(distance_set):
            for j,d_j in enumerate(temp_distance):
                if d_o == d_j:
                    trust_distance_order_matrix[i][j] = o

    source_pew_distance_matrix = np.load('E:/media-position/log/baseline/model/baseline_source_article.npy')
    source_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int32)
    for i,media_a in enumerate(data_map.dataset_list):
        temp_distance = source_pew_distance_matrix[i]
        distance_set = set(temp_distance)
        distance_set = sorted(list(distance_set))
        for o, d_o in enumerate(distance_set):
            for j,d_j in enumerate(temp_distance):
                if d_o == d_j:
                    source_distance_order_matrix[i][j] = o
    

    source_allsides_rank_similarity = 0
    for i in range(len(data_map.dataset_list)):
        # sort_distance += euclidean_distances(media_distance_order_matrix[i].reshape(1,-1), distance_order_matrix[i].reshape(1,-1))
        tau, p_value = kendalltau(source_pew_distance_matrix[i].reshape(1,-1), allsides_distance_order_matrix[i].reshape(1,-1))
        source_allsides_rank_similarity += tau
    source_allsides_rank_similarity /= len(data_map.dataset_list)

    trust_allsides_rank_similarity = 0
    for i in range(len(data_map.dataset_list)):
        # sort_distance += euclidean_distances(media_distance_order_matrix[i].reshape(1,-1), distance_order_matrix[i].reshape(1,-1))
        tau, p_value = kendalltau(trust_pew_distance_matrix[i].reshape(1,-1), allsides_distance_order_matrix[i].reshape(1,-1))
        trust_allsides_rank_similarity += tau
    trust_allsides_rank_similarity /= len(data_map.dataset_list)

    pew_rank_similarity = 0
    for i in range(len(data_map.dataset_list)):
        # sort_distance += euclidean_distances(media_distance_order_matrix[i].reshape(1,-1), distance_order_matrix[i].reshape(1,-1))
        tau, p_value = kendalltau(source_distance_order_matrix[i].reshape(1,-1), trust_distance_order_matrix[i].reshape(1,-1))
        pew_rank_similarity += tau
    pew_rank_similarity /= len(data_map.dataset_list)


    cluster_performance = _co_occurance_distance(source_cluster_list,trust_cluster_list)

    print('test')




def main():
    for file_list in [['en.valid'],['en.train'],['en.valid','en.train']]:
        # for method in ["tfidf","lda"]:
        for method in ["lda"]:
            for combine_method in ["average"]:
                get_baseline(['trust','source','mbr'], file_list, method, combine_method)
    # baseline_difference()

if __name__ == '__main__':
    main()