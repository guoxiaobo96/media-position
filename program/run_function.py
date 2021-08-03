from nltk import tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from numpy.lib.function_base import copy
from copy import deepcopy

from sklearn.feature_extraction.text import CountVectorizer
from .analysis import ClusterAnalysis, DistanceAnalysis, ClusterCompare
from .masked_token_util import MaskedTokenLabeller
from .baseline import BaselineCalculator
from .fine_tune_util import DataCollatorForLanguageModelingConsistency
from .data_augment_util import SelfDataAugmentor, CrossDataAugmentor
from .predict_util import MaksedPredictionDataset
from .ner_util import NERDataset
from .data import get_dataset, get_analysis_data, get_label_data, get_mask_score_data
from .model import MLMModel, SentenceReplacementModel, NERModel, ClassifyModel
from .config import BaselineArguments, DataArguments, DataAugArguments, FullArticleMap, MiscArgument, ModelArguments, TrainingArguments, AnalysisArguments, SourceMap, TrustMap, TwitterMap, ArticleMap, BaselineArticleMap
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import numpy as np
from matplotlib import pyplot as plt
from os import path
from typing import Dict, List
import json
import os
import joblib
import matplotlib
from transformers.utils.dummy_pt_objects import BertForSequenceClassification
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


# def label_masked_token(
#         misc_args: MiscArgument,
#         model_args: ModelArguments,
#         data_args: DataArguments,
#         training_args: TrainingArguments):
#     "This is the version which use BERT atttntion for masking labeled token"
#     original_sentence_list = list()

#     original_sentence_file = os.path.join(data_args.data_path, 'en.valid')
#     with open(original_sentence_file, mode='r', encoding='utf8') as fp:
#         for line in fp.readlines():
#             item = json.loads(line.strip())
#             original_sentence_list.append(item)

#     model = MaskedTokenLabeller(
#         misc_args, data_args, model_args, training_args)
    
#     if misc_args.global_debug:
#         original_sentence_list = original_sentence_list[:1000]

#     masked_sentence_list = list()
#     for item in tqdm(original_sentence_list):
#         label, probability, sentence_set = model.label_sentence(
#             item['sentence'])
#         if probability > 0.7 and label == item['label']:
#             masked_sentence_list.append({'original_sentence':item['sentence'],'masked_sentence':list(sentence_set)})
#     masked_sentence_file = os.path.join(data_args.data_path, 'en.masked')
#     with open(masked_sentence_file, mode='w',encoding='utf8') as fp:
#         for item in masked_sentence_list:
#             fp.write(json.dumps(item,ensure_ascii=False)+'\n')


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

    if data_args.label_method == 'trigram':
        stemmer = PorterStemmer()
        count_vectorize = CountVectorizer(min_df=5,ngram_range=(3,3),stop_words="english")
        tokenizer = count_vectorize.build_tokenizer()
        raw_data = list()

        for item in original_sentence_list:
            sentence = ' '.join([stemmer.stem(word) for word in tokenizer(item['sentence'])])
            raw_data.append(sentence)


        count_vectorize.fit_transform(raw_data)
        tokenizer = count_vectorize.build_tokenizer()

        for sentence in original_sentence_list:
            splited_sentence = tokenizer(sentence['sentence'])

        word_list = count_vectorize.get_feature_names()


        for sentence in tqdm(original_sentence_list):
            written_item = {"original_sentence":sentence["sentence"],"masked_sentence":list()}
            splited_sentence = tokenizer(sentence["sentence"])
            for index in range(len(splited_sentence)-2):
                phrase = ''
                count = 0
                while len(phrase.split(' '))<3:
                    if count+index < len(splited_sentence):
                        stemmed_word = stemmer.stem(splited_sentence[count+index])
                        phrase = phrase + ' '+ stemmed_word
                        phrase = phrase.strip()
                        count += 1
                    else:
                        break
                if len(phrase.split(' ')) < 3:
                    break
                if phrase in word_list:
                    full_phrase = splited_sentence[index:index+count]
                    for i in range(len(full_phrase)):
                        masked_phrase = deepcopy(full_phrase)
                        masked_phrase[i] = '[MASK]'
                        masked_sentence = splited_sentence[:index]+masked_phrase+splited_sentence[index+count:]
                        masked_sentence = ' '.join(masked_sentence)
                        written_item["masked_sentence"].append(masked_sentence)
                    masked_sentence_list.append(written_item)
    elif data_args.label_method == "bigram_outer":
        stemmer = PorterStemmer()
        count_vectorize = CountVectorizer(min_df=10,ngram_range=(2,2),stop_words="english")
        tokenizer = count_vectorize.build_tokenizer()
        raw_data = ['' for _ in range(10)]
        for item in tqdm(original_sentence_list,desc="load data"):
            sentence = ' '.join([stemmer.stem(word) for word in tokenizer(item['sentence'])])
            raw_data[item["label"]] = raw_data[item["label"]] + ' '+sentence
        count_vectorize.fit_transform(raw_data)
        for sentence in original_sentence_list:
            splited_sentence = tokenizer(sentence['sentence'])
        word_list = count_vectorize.get_feature_names()
        for sentence in tqdm(original_sentence_list):
            written_item = {"original_sentence":sentence["sentence"],"masked_sentence":list()}
            splited_sentence = tokenizer(sentence["sentence"])
            for index in range(len(splited_sentence)-1):
                phrase = ''
                count = 0
                while len(phrase.split(' '))<2:
                    if count+index < len(splited_sentence):
                        stemmed_word = stemmer.stem(splited_sentence[count+index])
                        phrase = phrase + ' '+ stemmed_word
                        phrase = phrase.strip()
                        count += 1
                    else:
                        break
                if len(phrase.split(' ')) < 2:
                    break
                if phrase in word_list:
                    full_phrase = splited_sentence[index:index+count]
                    pre_masked_sentence = None
                    post_masked_sentence = None
                    if index != 0:
                        pre_masked_sentence = splited_sentence[:index-1]+['[MASK]']+splited_sentence[index:]
                        pre_masked_sentence = ' '.join(pre_masked_sentence)
                        written_item["masked_sentence"].append(pre_masked_sentence)
                    if index + count <(len(splited_sentence)-1):
                        post_masked_sentence = splited_sentence[:index+count]+['[MASK]']+splited_sentence[index+count+1:]
                        post_masked_sentence = ' '.join(post_masked_sentence)
                        written_item["masked_sentence"].append(post_masked_sentence)
                    masked_sentence_list.append(written_item)

    elif data_args.label_method == 'bert':
        model = MaskedTokenLabeller(
            misc_args, data_args, model_args, training_args)
        for item in tqdm(original_sentence_list):
            label, probability, sentence_set = model.label_sentence(
                item['sentence'])
            if probability > 0.7 and label == item['label']:
                masked_sentence_list.append({'original_sentence':item['sentence'],'masked_sentence':list(sentence_set)})        

    masked_sentence_file = os.path.join(data_args.data_path, 'en.masked.'+data_args.label_method)
    with open(masked_sentence_file, mode='w',encoding='utf8') as fp:
        for item in masked_sentence_list:
            fp.write(json.dumps(item,ensure_ascii=False)+'\n')

def analysis(
    misc_args: MiscArgument,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    analysis_args: AnalysisArguments
) -> Dict:
    data_map = ArticleMap() if data_args.data_type == 'article' else TwitterMap()
    analysis_result = dict()
    model_list = dict()
    analysis_data = dict()
    analysis_data_temp = get_analysis_data(analysis_args)

    for k, v in analysis_data_temp.items():
        analysis_data[k] = dict()
        for d, _ in data_map.dataset_to_name.items():
            analysis_data[k][d] = v[d]

    analysis_data['concatenate.json'] = dict()
    analysis_data['average.json'] = dict()
    for k, v in analysis_data.items():
        for media, item in v.items():
            if media not in analysis_data['concatenate.json']:
                analysis_data['concatenate.json'][media] = dict()
            for w, c in item.items():
                if w not in analysis_data['concatenate.json'][media]:
                    analysis_data['concatenate.json'][media][w] = c
                else:
                    analysis_data['concatenate.json'][media][w] = float(
                        analysis_data['concatenate.json'][media][w]) + float(c)
    method = str()
    if analysis_args.analysis_compare_method == 'cluster':
        method = analysis_args.analysis_cluster_method
    elif analysis_args.analysis_compare_method == 'distance':
        method = analysis_args.analysis_distance_method
    for k, v in analysis_data.items():
        if k == 'average.json' or k == 'concatenate.json':
            continue
        if analysis_args.analysis_compare_method == 'cluster':
            analysis_model = ClusterAnalysis(
                misc_args, model_args, data_args, training_args, analysis_args)
        elif analysis_args.analysis_compare_method == 'distance':
            analysis_model = DistanceAnalysis(
                misc_args, model_args, data_args, training_args, analysis_args)
        model, cluster_result, dataset_list, encoded_list = analysis_model.analyze(
            v, k.split('.')[0], analysis_args)
        analysis_result[k] = cluster_result
        model_list[k] = model
        for i, encoded_data in enumerate(encoded_list):
            if dataset_list[i] not in analysis_data['average.json']:
                analysis_data['average.json'][dataset_list[i]] = list()
            analysis_data['average.json'][dataset_list[i]].append(encoded_data)
    average_distance_matrix = np.zeros(
        (len(data_map.dataset_list), len(data_map.dataset_list)))

    for i, dataset_name_a in enumerate(data_map.dataset_list):
        for j, dataset_name_b in enumerate(data_map.dataset_list):
            if i == j:
                continue
            average_distance = 0
            encoded_a = analysis_data['average.json'][dataset_name_a]
            encoded_b = analysis_data['average.json'][dataset_name_b]
            for k in range(len(analysis_data) - 2):
                average_distance += euclidean_distances(
                    encoded_a[k].reshape(1, -1), encoded_b[k].reshape(1, -1))[0][0]
            average_distance_matrix[i][j] = average_distance
    analysis_data['average.json'] = average_distance_matrix
    model, cluster_result, _, _ = analysis_model.analyze(
        analysis_data['average.json'], 'average', analysis_args, encode=False, dataset_list=list(data_map.dataset_list))
    model_list['average.json'] = model
    analysis_result['average.json'] = cluster_result
    conclusion = dict()
    # for k, v in analysis_result.items():
    #     analysis_file = os.path.join(analysis_args.analysis_result_dir, k.split('.')[0])
    #     with open(analysis_file, mode='a',encoding='utf8') as fp:
    #         fp.write(json.dumps({'encode': analysis_args.analysis_encode_method,'method':method, 'result':v},ensure_ascii=False)+'\n')
    #     for country, distance in v.items():
    #         if country not in conclusion:
    #             conclusion[country] = dict()
    #         conclusion[country][k] = distance
    if analysis_args.analysis_compare_method == 'distance':
        for k, v in analysis_result.items():
            label_list, data = v
            _draw_heatmap(data, label_list, label_list)
            # data = pd.DataFrame(v,columns=k,index=k)
            # sns.heatmap(data)
            plt_file = os.path.join(analysis_args.analysis_result_dir,
                                    analysis_args.analysis_encode_method+'_'+method+'_' + k.split('.')[0]+'.png')
            plt.savefig(plt_file, bbox_inches='tight')
            plt.close()
        # with open(os.path.join(analysis_args.analysis_result_dir, analysis_args.analysis_encode_method+'_'+method+'.csv'), mode='w',encoding='utf8') as fp:
        #     title = 'country,'
        #     for i in range(len(analysis_result)):
        #         title = title + str(i+1)+','
        #     fp.write(title+'\n')
        #     for country, distance_list in conclusion.items():
        #         record = country+','
        #         for i in range(len(distance_list)):
        #             record = record+str(distance_list[str(i+1)+'.json'])+','
        #         fp.write(record+'\n')
    else:
        base_model = joblib.load(
            'log/baseline/model/baseline_trust_'+data_args.data_type+'.c')
        model_list['base'] = base_model
        base_model = joblib.load(
            'log/baseline/model/baseline_source_'+data_args.data_type+'.c')
        model_list['distance_base'] = base_model
        cluster_compare = ClusterCompare(misc_args, analysis_args)

        label_list = []
        for name in data_map.dataset_list:
            if name in data_map.left_dataset_list:
                label_list.append(1)
            else:
                label_list.append(0)

        analysis_result = cluster_compare.compare(model_list)
        analysis_result = sorted(analysis_result.items(), key=lambda x: x[1])
        analysis_result = {k: v for k, v in analysis_result}

        result_path = os.path.join(
            analysis_args.analysis_result_dir, analysis_args.graph_distance)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        result_file = os.path.join(
            result_path, analysis_args.analysis_encode_method+'_'+method+'_'+analysis_args.graph_kernel+'.txt')
        with open(result_file, mode='w', encoding='utf8') as fp:
            for k, v in analysis_result.items():
                fp.write(k+' : '+str(v)+'\n')
    return analysis_result


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
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = os.path.join(os.path.join(os.path.join(log_dir, training_args.loss_type),data_args.label_method),data_args.data_type)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(os.path.join(log_dir, 'json'))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, data_args.dataset+'.json')

    batched_masked_sentence_list: List = list()
    masked_sentence_list = list()
    masked_sentence_dict = dict()

    masked_sentence_file = os.path.join(os.path.join(os.path.join(data_args.data_dir,'all'),'original'),'en.masked.'+data_args.label_method)
    with open(masked_sentence_file,mode='r',encoding='utf8') as fp:
        for line in fp.readlines():
            item = json.loads(line.strip())
            original_sentence = item['original_sentence']
            masked_sentences = item['masked_sentence']
            masked_sentence_list.extend(masked_sentences)
            for masked_sentence in masked_sentences:
                masked_sentence_dict[masked_sentence] = original_sentence

            
    batch_size = 64
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

        splited_sentece = sentence.replace('.','').split(' ')
        for i, token in enumerate(splited_sentece):
            if '[MASK]' in token:
                masked_index = i

        record_dict[original_sentence]['word'][masked_index] = dict()
        for item in items:
            record_dict[original_sentence]['word'][masked_index][item["token_str"]] = str(
                round(item["score"], 3))
            word_set.add(item["token_str"])

    with open(log_file, mode='a', encoding='utf8') as fp:
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
    base_line: str
) -> Dict:
    data_map = BaselineArticleMap()
    analysis_result = dict()
    model_list = dict()
    analysis_data = dict()
    sentence_position_data = dict()

    if not os.path.exists(analysis_args.analysis_result_dir):
        os.makedirs(analysis_args.analysis_result_dir)

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

    # analysis_data['concatenate'] = dict()
    # for k, v in analysis_data.items():
    #     for media, item in v.items():
    #         if media not in analysis_data['concatenate']:
    #             analysis_data['concatenate'][media] = dict()
    #         for w, c in item.items():
    #             if w not in analysis_data['concatenate'][media]:
    #                 analysis_data['concatenate'][media][w] = c
    #             else:
    #                 analysis_data['concatenate'][media][w] = float(analysis_data['concatenate'][media][w]) + float(c)

    method = str()
    if analysis_args.analysis_compare_method == 'cluster':
        method = analysis_args.analysis_cluster_method
        analysis_model = ClusterAnalysis(
            misc_args, model_args, data_args, training_args, analysis_args)
        precomputed_analysis_model = ClusterAnalysis(
            misc_args, model_args, data_args, training_args, analysis_args,pre_computer=True)
    elif analysis_args.analysis_compare_method == 'distance':
        method = analysis_args.analysis_distance_method
        analysis_model = DistanceAnalysis(
            misc_args, model_args, data_args, training_args, analysis_args)
    for k, v in tqdm(analysis_data.items(), desc="Build cluster"):
        if k == 'media_average' or k == 'concatenate':
            continue
        try:
            model, cluster_result, dataset_list, encoded_list = analysis_model.analyze(
                v, str(k), analysis_args, keep_result=False)
            analysis_result[k] = cluster_result
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

    conclusion = dict()
    # for k, v in analysis_result.items():
    #     analysis_file = os.path.join(analysis_args.analysis_result_dir, k.split('.')[0])
    #     with open(analysis_file, mode='a',encoding='utf8') as fp:
    #         fp.write(json.dumps({'encode': analysis_args.analysis_encode_method,'method':method, 'result':v},ensure_ascii=False)+'\n')
    #     for country, distance in v.items():
    #         if country not in conclusion:
    #             conclusion[country] = dict()
    #         conclusion[country][k] = distance

    if analysis_args.analysis_compare_method == 'distance':
        for k, v in analysis_result.items():
            label_list, data = v
            _draw_heatmap(data, label_list, label_list)
            plt_file = os.path.join(analysis_args.analysis_result_dir,
                                    analysis_args.analysis_encode_method+'_'+method+'_' + k.split('.')[0]+'.png')
            plt.savefig(plt_file, bbox_inches='tight')
            plt.close()
    else:
        if base_line == 'source':
            distance_base = 'trust'
        else:
            distance_base = 'source'

        base_model = joblib.load('log/baseline/model/baseline_'+base_line+'_article.c')
        model_list['base'] = base_model
        base_model = joblib.load(
            'log/baseline/model/baseline_'+distance_base+'_article.c')
        model_list['distance_base'] = base_model
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
                        average_distance += cosine_distances(
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
        sentence_position_data['distance_base'] = {
            'sentence': 'distance_base', 'position': -2, 'word': 'distance_base'}

        result = dict()
        average_distance = dict()
        for k, v in tqdm(analysis_result, desc="Combine cluster analyze"):
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

        result_path = os.path.join(
            analysis_args.analysis_result_dir, analysis_args.graph_distance)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        result_file = os.path.join(result_path, analysis_args.analysis_encode_method +
                                   '_'+method+'_'+analysis_args.graph_kernel+'_sort_'+base_line+'.json')
        with open(result_file, mode='w', encoding='utf8') as fp:
            for k, v in analysis_result.items():
                fp.write(json.dumps(v, ensure_ascii=False)+'\n')

        result_file = os.path.join(result_path, analysis_args.analysis_encode_method +
                                   '_'+method+'_'+analysis_args.graph_kernel+'_sentence_'+base_line+'.json')
        with open(result_file, mode='w', encoding='utf8') as fp:
            for k, v in result.items():
                v['sentence'] = k
                fp.write(json.dumps(v, ensure_ascii=False)+'\n')

    print("The basic distance is {}".format(result['distance_base'][-2][0]))
    print("The cluster average performance is {}".format(
        result['cluster_average'][-2][0]))
    print("The media average performance is {}".format(
        result['media_average'][-2][0]))

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

    # Loop over data dimensions and create text annotations.
    # for i in range(len(x_list)):
    #     for j in range(len(y_list)):
    #         text = ax.text(j, i, data[i, j],
    #                     ha="center", va="center", color="w")
    # ax.set_title("Harvest of local farmers (in tons/year)")
    # fig.tight_layout()
    # plt.show()
