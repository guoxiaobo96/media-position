from .config import BaselineArguments, DataArguments, DataAugArguments, FullArticleMap, MiscArgument, ModelArguments, TrainingArguments, AnalysisArguments, SourceMap, TrustMap, TwitterMap, ArticleMap, BaselineArticleMap
from .model import MLMModel, SentenceReplacementModel, NERModel, ClassifyModel
from .data import get_dataset, get_analysis_data, get_label_data, get_mask_score_data
from .ner_util import NERDataset
from .predict_util import MaksedPredictionDataset
from .data_augment_util import SelfDataAugmentor, CrossDataAugmentor
from .fine_tune_util import DataCollatorForLanguageModelingConsistency
from .baseline import BaselineCalculator
from .masked_token_util import MaskedTokenLabeller, ngram_inner_label, ngram_outer_label, random_label
from .analysis import ClusterAnalysis, DistanceAnalysis, ClusterCompare, CorrelationAnalysis, CorrelationCompare
from nltk import tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from numpy.lib.function_base import copy
from copy import deepcopy
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, cosine_distances, manhattan_distances
from scipy.stats import kendalltau
import numpy as np
from matplotlib import pyplot as plt, transforms
from os import path
from typing import Dict, List
import json
import os
import joblib
import matplotlib
import transformers

matplotlib.use('Agg')


def generate_baseline(
    misc_args: MiscArgument,
    baseline_args: BaselineArguments,
    data_args: DataArguments
) -> Dict:
    baseline_calculator = BaselineCalculator(
        misc_args, baseline_args, data_args)
    baseline_calculator.load_data()
    baseline_calculator.encode_data()
    result = baseline_calculator.feature_analysis()
    return result


def train_lm(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Dict:
    model = MLMModel(model_args, data_args, training_args)
    train_dataset = (
        get_dataset(training_args, data_args, model_args, tokenizer=model.tokenizer,
                    cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(training_args, data_args, model_args, tokenizer=model.tokenizer,
                    evaluate=True, cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    model.train(train_dataset, eval_dataset)


def eval_lm(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Dict:
    model = MLMModel(model_args, data_args, training_args)
    eval_dataset = (
        get_dataset(training_args, data_args, tokenizer=model.tokenizer,
                    evaluate=True, cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    record_file = os.path.join(data_args.data_dir.split(
        '_')[-1].split('/')[0], data_args.dataset)
    model.eval(eval_dataset, record_file, verbose=False)


def train_classifier(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Dict:
    model = ClassifyModel(model_args, data_args, training_args)
    train_dataset = (
        get_dataset(training_args, data_args, model_args, tokenizer=model.tokenizer,
                    cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(training_args, data_args, model_args, tokenizer=model.tokenizer,
                    evaluate=True, cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    model.train(train_dataset, eval_dataset)


def label_masked_token(
        misc_args: MiscArgument,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments):
    original_sentence_list = list()

    original_sentence_file = os.path.join(data_args.data_path, 'en.valid')
    with open(original_sentence_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            item = json.loads(line.strip())
            original_sentence_list.append(item)
    masked_sentence_list = list()

    if misc_args.global_debug:
        original_sentence_list = original_sentence_list[:1000]

    if data_args.label_method == 'trigram_inner':
        masked_sentence_list = ngram_inner_label(
            original_sentence_list, n_gram=3, min_df=5)
    elif data_args.label_method == 'bigram_inner':
        masked_sentence_list = ngram_inner_label(
            original_sentence_list, n_gram=2, min_df=5)
    elif data_args.label_method == "bigram_outer":
        masked_sentence_list = ngram_outer_label(
            original_sentence_list, n_gram=2, min_df=10)
    elif data_args.label_method == 'bert':
        model = MaskedTokenLabeller(
            misc_args, data_args, model_args, training_args)
        for item in tqdm(original_sentence_list):
            if item['sentence'] == '':
                continue
            label, probability, sentence_set = model.label_sentence(
                item['sentence'])
            if probability > 0.7 and label == item['label']:
                masked_sentence_list.append(
                    {'original_sentence': item['sentence'], 'masked_sentence': list(sentence_set)})
    elif data_args.label_method == 'random':
        masked_sentence_list = random_label(original_sentence_list, 0.5, 0.2)
    masked_sentence_file = os.path.join(
        data_args.data_path, 'en.masked.'+data_args.label_method)
    with open(masked_sentence_file, mode='w', encoding='utf8') as fp:
        for item in masked_sentence_list:
            fp.write(json.dumps(item, ensure_ascii=False)+'\n')


def label_score_predict(

    misc_args: MiscArgument,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Dict:
    dataset_map = FullArticleMap()

    data_type = list()
    dataset = data_args.dataset

    data_type.append('dataset')
    if dataset in ['vanilla']:
        data_type = ['dataset', 'position']

    model = MLMModel(model_args, data_args, training_args, vanilla_model=True)
    word_set = set()

    log_dir = os.path.join(
        misc_args.log_dir, data_args.data_dir.split('_')[1].split('/')[0])
    log_dir = os.path.join(log_dir,str(training_args.seed))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = os.path.join(os.path.join(os.path.join(
        log_dir, training_args.loss_type), data_args.label_method), data_args.data_type)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(os.path.join(log_dir, 'json'))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, data_args.dataset+'.json')

    batched_masked_sentence_list: List = list()
    masked_sentence_list = list()
    masked_sentence_dict = dict()

    masked_sentence_file = os.path.join(os.path.join(os.path.join(
        data_args.data_dir, 'all'), 'original'), 'en.masked.'+data_args.label_method)
    with open(masked_sentence_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            item = json.loads(line.strip())
            original_sentence = item['original_sentence']
            masked_sentences = item['masked_sentence']
            masked_sentence_list.extend(masked_sentences)
            for masked_sentence in masked_sentences:
                masked_sentence_dict[masked_sentence] = original_sentence

    batch_size = 32
    index = 0
    while (index < len(masked_sentence_list)):
        batched_masked_sentence_list.append(
            masked_sentence_list[index:index+batch_size])
        index += batch_size

    if misc_args.global_debug:
        batched_masked_sentence_list = batched_masked_sentence_list[:10]

    results = dict()
    for batch_sentence in tqdm(batched_masked_sentence_list):
        result = model.predict(batch_sentence)
        results.update(result)

    record_dict = dict()

    for sentence, items in results.items():
        original_sentence = masked_sentence_dict[sentence]
        if original_sentence not in record_dict:
            record_dict[original_sentence] = {
                'sentence': original_sentence, 'word': dict()}

        splited_sentece = sentence.replace('.', '').split(' ')
        for i, token in enumerate(splited_sentece):
            if '[MASK]' in token:
                masked_index = i

        record_dict[original_sentence]['word'][masked_index] = dict()
        for item in items:
            record_dict[original_sentence]['word'][masked_index][item["token_str"]] = str(
                round(item["score"], 3))
            word_set.add(item["token_str"])

    with open(log_file, mode='w', encoding='utf8') as fp:
        for _, item in record_dict.items():
            fp.write(json.dumps(item, ensure_ascii=False)+'\n')

    dict_file = os.path.join(os.path.join(log_dir, 'dict'), 'word_set.txt')
    if not os.path.exists(os.path.join(log_dir, 'dict')):
        os.makedirs(os.path.join(log_dir, 'dict'))
    if os.path.exists(dict_file):
        with open(dict_file, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                word_set.add(line.strip())
    with open(dict_file, mode='w', encoding='utf8') as fp:
        for token in word_set:
            fp.write(token+'\n')

def label_score_analysis(
    misc_args: MiscArgument,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    analysis_args: AnalysisArguments,
    ground_truth: str
) -> Dict:
    data_map = BaselineArticleMap()
<<<<<<< HEAD
    ground_truth_distance_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.float32)
    bias_distance_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)))
    allsides_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int)
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
                    allsides_distance_order_matrix[i][j] = o
=======
    ground_truth_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int)
    if ground_truth == "MBR":
        ground_truth_distance_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.float32)
        bias_distance_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)))
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
    elif ground_truth in ["source",'trust']:
        ground_truth_model = joblib.load(
        './log/baseline/model/baseline_'+ground_truth+'_article.c')
        ground_truth_distance_matrix = np.load('./log/baseline/model/baseline_'+ground_truth+'_article.npy')
        for i,media_a in enumerate(data_map.dataset_list):
            temp_distance = ground_truth_distance_matrix[i]
            distance_set = set(temp_distance)
            distance_set = sorted(list(distance_set))
            for o, d_o in enumerate(distance_set):
                for j,d_j in enumerate(temp_distance):
                    if d_o == d_j:
                        ground_truth_distance_order_matrix[i][j] = o
>>>>>>> 275ecfefe8664a60ff8fe75eb19166375f08ee27
                    



    analysis_result = dict()
    model_list = dict()
    analysis_data = dict()
    sentence_position_data = dict()

    if not os.path.exists(analysis_args.analysis_result_dir):
        os.makedirs(analysis_args.analysis_result_dir)
    analysis_record_file = '/'.join(analysis_args.analysis_result_dir.split('/')[:6])
    if not os.path.exists(analysis_record_file):
        os.makedirs(analysis_record_file)
    analysis_record_file = os.path.join(analysis_record_file,'record')

    error_count = 0
    analysis_data_temp = get_label_data(misc_args, analysis_args, data_args)
    index = 0
    for k, item in tqdm(analysis_data_temp.items(), desc="Load data"):
        for position, v in item.items():
            if len(v) != len(data_map.dataset_list):
                continue
            try:
                sentence_position_data[index] = {
                    'sentence': k, 'position': position, 'word': k.split(' ')[int(position)]}
                analysis_data[index] = dict()
                for dataset in data_map.dataset_list:
                    analysis_data[index][dataset] = v[dataset]
                index += 1
            except (IndexError, KeyError):
                length = len(k.split(' '))
                error_count += 1
                continue
        if misc_args.global_debug and index > 100:
            break
    analysis_data['media_average'] = dict()

    method = str()
    if analysis_args.analysis_compare_method == 'cluster':
        method = analysis_args.analysis_cluster_method
        analysis_model = ClusterAnalysis(
            misc_args, model_args, data_args, training_args, analysis_args)
        precomputed_analysis_model = ClusterAnalysis(
            misc_args, model_args, data_args, training_args, analysis_args, pre_computer=True)
    elif analysis_args.analysis_compare_method == 'distance':
        method = analysis_args.analysis_distance_method
        analysis_model = DistanceAnalysis(
            misc_args, model_args, data_args, training_args, analysis_args)
    elif analysis_args.analysis_compare_method == 'correlation':
        method = analysis_args.analysis_correlation_method
        analysis_model = CorrelationAnalysis(
            misc_args, model_args, data_args, training_args, analysis_args)

    for k, v in tqdm(analysis_data.items(), desc="encode instance"):
        if k == 'media_average' or k == 'concatenate':
            continue
        try:
            model, result, dataset_list, encoded_list = analysis_model.analyze(
                v, str(k), analysis_args, keep_result=False,data_map=data_map)
            if model is None:
                continue
            analysis_result[k] = result
            model_list[k] = model
            for i, encoded_data in enumerate(encoded_list):
                if dataset_list[i] not in analysis_data['media_average']:
                    analysis_data['media_average'][dataset_list[i]] = list()
                analysis_data['media_average'][dataset_list[i]].append(
                    encoded_data)
        except ValueError:
            continue
    average_distance_matrix = np.zeros(
        (len(data_map.dataset_list), len(data_map.dataset_list)))
    performance = 0
    if analysis_args.analysis_compare_method == 'distance':
        for k, v in analysis_result.items():
            label_list, data = v
            _draw_heatmap(data, label_list, label_list)
            plt_file = os.path.join(analysis_args.analysis_result_dir,
                                    analysis_args.analysis_encode_method+'_'+method+'_' + k.split('.')[0]+'.png')
            plt.savefig(plt_file, bbox_inches='tight')
            plt.close()
    elif analysis_args.analysis_compare_method == 'cluster':
        model_list['base'] = ground_truth_model
        cluster_compare = ClusterCompare(misc_args, analysis_args)
        analysis_result = cluster_compare.compare(model_list)

        for i, dataset_name_a in enumerate(tqdm(data_map.dataset_list, desc="Combine cluster")):
            for j, dataset_name_b in enumerate(data_map.dataset_list):
                if i == j or average_distance_matrix[i][j] != 0:
                    continue
                average_distance = 0
                encoded_a = analysis_data['media_average'][dataset_name_a]
                encoded_b = analysis_data['media_average'][dataset_name_b]
                for k in range(len(encoded_a)):
                    if k in analysis_result and (analysis_result[k] < analysis_args.analysis_threshold or analysis_args.analysis_threshold == -1):
                        # average_distance += cosine_distances(
                        #     encoded_a[k].reshape(1, -1), encoded_b[k].reshape(1, -1))[0][0]
                        average_distance += euclidean_distances(
                            encoded_a[k].reshape(1, -1), encoded_b[k].reshape(1, -1))[0][0]
                average_distance_matrix[i][j] = average_distance / \
                    len(encoded_a)
                average_distance_matrix[j][i] = average_distance / \
                    len(encoded_a)
        analysis_data['media_average'] = average_distance_matrix

        model, cluster_result, _, _ = precomputed_analysis_model.analyze(
            analysis_data['media_average'], 'media_average', analysis_args, encode=False, dataset_list=list(data_map.dataset_list))
        model_list['media_average'] = model

        cluster_average = list()
        for _, v in analysis_result.items():
            if v < analysis_args.analysis_threshold or analysis_args.analysis_threshold == -1:
                cluster_average.append(v)

        analysis_result = cluster_compare.compare(model_list)
        analysis_result['cluster_average'] = np.mean(cluster_average)
        analysis_result = sorted(analysis_result.items(), key=lambda x: x[1])
        sentence_position_data['media_average'] = {
            'sentence': 'media_average', 'position': -2, 'word': 'media_average'}
        sentence_position_data['cluster_average'] = {
            'sentence': 'cluster_average', 'position': -2, 'word': 'cluster_average'}

        media_distance = analysis_data["media_average"]
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
    elif analysis_args.analysis_compare_method == 'correlation':
        model_list['base'] = ground_truth_distance_order_matrix
        compare_model = CorrelationCompare(misc_args, analysis_args)
        analysis_result = compare_model.compare(model_list)
        

    for i, dataset_name_a in enumerate(tqdm(data_map.dataset_list, desc="Combine cluster")):
        for j, dataset_name_b in enumerate(data_map.dataset_list):
            if i == j or average_distance_matrix[i][j] != 0:
                continue
            average_distance = 0
            encoded_a = analysis_data['media_average'][dataset_name_a]
            encoded_b = analysis_data['media_average'][dataset_name_b]
            for k in range(len(encoded_a)):
                if k in analysis_result and (analysis_result[k] < analysis_args.analysis_threshold or analysis_args.analysis_threshold == -1):
                    # average_distance += cosine_distances(
                    #     encoded_a[k].reshape(1, -1), encoded_b[k].reshape(1, -1))[0][0]
                    average_distance += euclidean_distances(
                        encoded_a[k].reshape(1, -1), encoded_b[k].reshape(1, -1))[0][0]
            average_distance_matrix[i][j] = average_distance / \
                len(encoded_a)
            average_distance_matrix[j][i] = average_distance / \
                len(encoded_a)
    analysis_data['media_average'] = average_distance_matrix

    performance_average = list()
    for _, v in analysis_result.items():
        if (v < analysis_args.analysis_threshold or analysis_args.analysis_threshold == -1):
            performance_average.append(v)
    analysis_result['performance_average'] = np.nanmean(performance_average)
    analysis_result = sorted(analysis_result.items(), key=lambda x: x[1])
    sentence_position_data['media_average'] = {
        'sentence': 'media_average', 'position': -2, 'word': 'media_average'}
    sentence_position_data['performance_average'] = {
        'sentence': 'performance_average', 'position': -2, 'word': 'performance_average'}

    media_distance = analysis_data["media_average"]
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



    if analysis_args.analysis_compare_method == 'distance':
        pass
    elif analysis_args.analysis_compare_method == 'cluster':
        model, cluster_result, _, _ = precomputed_analysis_model.analyze(
            analysis_data['media_average'], 'media_average', analysis_args, encode=False, dataset_list=list(data_map.dataset_list))
        model_list['media_average'] = model

        cluster_average = list()
        for _, v in analysis_result.items():
            if v < analysis_args.analysis_threshold or analysis_args.analysis_threshold == -1:
                cluster_average.append(v)

        analysis_result = cluster_compare.compare(model_list)
        analysis_result['cluster_average'] = np.mean(cluster_average)
        analysis_result = sorted(analysis_result.items(), key=lambda x: x[1])
        sentence_position_data['media_average'] = {
            'sentence': 'media_average', 'position': -2, 'word': 'media_average'}
        sentence_position_data['cluster_average'] = {
            'sentence': 'cluster_average', 'position': -2, 'word': 'cluster_average'}

        media_distance = analysis_data["media_average"]
        media_distance_order_matrix = np.zeros(shape=(len(data_map.dataset_bias),len(data_map.dataset_bias)),dtype=np.int)
    elif analysis_args.analysis_compare_method == 'correlation':
        if  method == 'tau':
            for i in range(len(data_map.dataset_list)):
                tau, p_value = kendalltau(media_distance_order_matrix[i].reshape(1,-1), ground_truth_distance_order_matrix[i].reshape(1,-1))
                performance += tau
            performance /= len(data_map.dataset_list)
        elif method == 'pearson':
            for i,media_a in enumerate(data_map.dataset_list):
                pearson = np.corrcoef(ground_truth_distance_matrix[i].reshape(1,-1),media_distance[i].reshape(1,-1))
                performance += pearson[0][1]
            performance /= len(data_map.dataset_list)

        step_size = 0.05
        x_list = np.arange(-1,1+step_size,step_size)
        distribution = [0 for _ in range(len(x_list))]
        for p in performance_average:
            try:
                distribution[int(p/step_size) + 20] += 1
            except ValueError:
                continue

        distribution = np.array(distribution) / np.sum(distribution)
        mean_performance = np.nanmean(performance_average)
        median_performance = np.nanmedian(performance_average)

        start_index = 0
        end_index = 0
        for i, v in enumerate(distribution):
            if v != 0:
                end_index = i
                if start_index == 0:
                    start_index = i
        
        result_path = os.path.join(analysis_args.analysis_result_dir,method)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        sort_result_file = os.path.join(result_path, analysis_args.analysis_encode_method +'_sort_'+ground_truth+'.json')

        sentence_result_file = os.path.join(result_path, analysis_args.analysis_encode_method +'_sentence_'+ground_truth+'.json')
        plt.title('Distribution of Tau ', fontsize=20)
        plt_file = os.path.join(result_path, data_args.dataset+'_'+analysis_args.analysis_encode_method +'_distribution_'+ground_truth+'.jpg')
        plt.plot(x_list[start_index:end_index+1], distribution[start_index:end_index+1],linewidth=2)
        plt.xticks(fontsize=20)
        plt.xlabel("Kendall rank correlation coefficients", fontsize=20)
        plt.ylabel("Percent of Instances", fontsize=20)

        # plt.vlines(performance,0,np.max(distribution)+0.1, colors='r', label="The model performance is "+ str(round(performance,2)))
        # plt.vlines(mean_performance,0,np.max(distribution)+0.1, colors='g', label="The mean performance is "+ str(round(mean_performance,2)))
        # plt.vlines(median_performance,0,np.max(distribution)+0.1, colors='b', label="The median performance is "+ str(round(median_performance,2)))

        plt.savefig(plt_file,bbox_inches='tight')
        plt.close()




    result = dict()
    average_distance = dict()
    for k, v in tqdm(analysis_result, desc="Combine analyze"):
        sentence = sentence_position_data[k]['sentence']
        position = sentence_position_data[k]['position']
        word = sentence_position_data[k]['word']
        if sentence not in result:
            average_distance[sentence] = list()
            result[sentence] = dict()
        result[sentence][position] = (v, word)
        average_distance[sentence].append(v)

    for sentence, average_distance in average_distance.items():
        result[sentence][-1] = (np.mean(average_distance),
                                'sentence_average')

    sentence_list = list(result.keys())
    analysis_result = {k: {'score': v, 'sentence': sentence_list.index(
        sentence_position_data[k]['sentence'])+1, 'position': sentence_position_data[k]['position'], 'word': sentence_position_data[k]['word']} for k, v in analysis_result}


    with open(sort_result_file, mode='w', encoding='utf8') as fp:
        for k, v in analysis_result.items():
            fp.write(json.dumps(v, ensure_ascii=False)+'\n')

    with open(sentence_result_file, mode='w', encoding='utf8') as fp:
        for k, v in result.items():
            v['sentence'] = k
            fp.write(json.dumps(v, ensure_ascii=False)+'\n')

    record_item = {'ground_truth':ground_truth,'augmentation_method':data_args.data_type.split('/')[0],'analysis_compare_method':analysis_args.analysis_compare_method,'method':method,'performance':round(performance,2)}
    with open(analysis_record_file,mode='a',encoding='utf8') as fp:
        fp.write(json.dumps(record_item,ensure_ascii=False)+'\n')
    print("The performance on {} is {}".format(ground_truth, round(performance,2)))

    print("Analysis finish")
    return analysis_result


def data_augemnt(
    misc_args: MiscArgument,
    data_args: DataArguments,
    aug_args: DataAugArguments
):
    if aug_args.augment_type in ['duplicate', 'sentence_order_replacement', 'span_cutoff', 'word_order_replacement', 'word_replacement', 'sentence_replacement', 'combine_aug']:
        data_augmentor = SelfDataAugmentor(misc_args, data_args, aug_args)
    elif aug_args.augment_type in ['cross_sentence_replacement']:
        data_augmentor = CrossDataAugmentor(misc_args, data_args, aug_args)
    data_augmentor.data_augment(aug_args.augment_type)
    data_augmentor.save()


def train_mask_score_model(model_args: ModelArguments,
                           data_args: DataArguments,
                           training_args: TrainingArguments,
                           analysis_args: AnalysisArguments) -> None:

    model = NERModel(model_args, data_args, training_args)

    train_dataset, eval_dataset = get_mask_score_data(
        analysis_args, data_args, model.tokenizer)
    model.train(train_dataset, eval_dataset)


def _draw_heatmap(data, x_list, y_list):
    fig, ax = plt.subplots()
    im = ax.imshow(data)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("", rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_list)))
    ax.set_yticks(np.arange(len(y_list)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_list)
    ax.set_yticklabels(y_list)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

def encode_media(
    misc_args: MiscArgument,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Dict:
    model = MLMModel(model_args, data_args, training_args)
    train_dataset = (
        get_dataset(training_args, data_args, model_args, tokenizer=model.tokenizer,
                    cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    if training_args.do_train:
        model.train(train_dataset, train_dataset)

    model._model = transformers.BertModel.from_pretrained(training_args.output_dir,
                    from_tf=bool(".ckpt" in training_args.output_dir),
                    config=model._config,
                    cache_dir=model._model_args.cache_dir,
                )

    batch_size = 32
    index = 0
    sentence_list = list()
    batched_sentence_list = list()
    with open(data_args.train_data_file,mode='r',encoding='utf8') as fp:
        for line in fp.readlines():
            item = json.loads(line.strip())
            if len(item['original']) > 500:
                item['original'] = item['original'][:500]
            sentence_list.append(item['original'])
        

    while (index < len(sentence_list)):
        batched_sentence_list.append(sentence_list[index:index+batch_size])
        index += batch_size

    results = dict()
    for batch_sentence in tqdm(batched_sentence_list):
        result = model.encode(batch_sentence)
        results.update(result)
    encode_result = list()
    for _, v in results.items():
        encode_result.append(v)
    encode_result = np.array(encode_result)
    encode_result = encode_result.mean(axis=0)
    saved_file = os.path.join(os.path.join(misc_args.log_dir),data_args.dataset)
    if not os.path.exists(saved_file):
        os.makedirs(saved_file)
    saved_file = os.path.join(saved_file,training_args.loss_type+'.npy')
    np.save(saved_file,encode_result)