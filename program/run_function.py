from os import path
from posix import listdir
from typing import Dict, List
import json
import os
import joblib
import matplotlib
from tensorflow.python.util.tf_decorator import make_decorator
from transformers.file_utils import SENTENCEPIECE_IMPORT_ERROR
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm


from .config import DataArguments, MiscArgument, ModelArguments, TrainingArguments, AdapterArguments, AnalysisArguments, SourceMap, TrustMap, TwitterMap, ArticleMap
from .model import MLMAdapterModel, SentenceReplacementAdapterModel, NERModel
from .data import get_dataset, get_analysis_data, get_label_data, get_mask_score_data
from .analysis import ClusterAnalysis,DistanceAnalysis,ClusterCompare
from .ner_util import NERDataset


def mlm_train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    adapter_args: AdapterArguments
) -> Dict:
    model = MLMAdapterModel(model_args, data_args, training_args, adapter_args)
    train_dataset = (
        get_dataset(data_args, tokenizer=model.tokenizer,
                    cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=model.tokenizer,
                    evaluate=True, cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    model.train(train_dataset, eval_dataset)

def mlm_eval(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    adapter_args: AdapterArguments
) -> Dict:
    model = MLMAdapterModel(model_args, data_args, training_args, adapter_args)
    eval_dataset = (
        get_dataset(data_args, tokenizer=model.tokenizer,
                    evaluate=True, cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    record_file = data_args.data_dir.split('_')[-1].split('/')[0]+'_'+data_args.dataset+'_'
    model.eval(eval_dataset, record_file)

def sentence_replacement_train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    adapter_args: AdapterArguments
) -> Dict:
    model = SentenceReplacementAdapterModel(model_args, data_args, training_args, adapter_args)
    train_dataset, eval_dataset, number_label = get_dataset(data_args, model.tokenizer)
    model.train(train_dataset, eval_dataset, number_label)

def analysis(
    misc_args: MiscArgument,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    analysis_args: AnalysisArguments
) -> Dict:
    data_map = ArticleMap () if data_args.data_type == 'article' else TwitterMap()
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
                    analysis_data['concatenate.json'][media][w] = float(analysis_data['concatenate.json'][media][w]) + float(c)
    method = str()
    if analysis_args.analysis_compare_method == 'cluster':
        method = analysis_args.analysis_cluster_method
    elif analysis_args.analysis_compare_method == 'distance':
        method = analysis_args.analysis_distance_method
    for k, v in analysis_data.items():
        if k == 'average.json' or k == 'concatenate.json':
            continue
        if analysis_args.analysis_compare_method == 'cluster':
            analysis_model = ClusterAnalysis(misc_args, model_args, data_args, training_args, analysis_args)
        elif analysis_args.analysis_compare_method == 'distance':
            analysis_model = DistanceAnalysis(misc_args, model_args, data_args, training_args, analysis_args)      
        model, cluster_result, dataset_list, encoded_list = analysis_model.analyze(v, k.split('.')[0], analysis_args)
        analysis_result[k] = cluster_result
        model_list[k] = model
        for i, encoded_data in enumerate(encoded_list):
            if dataset_list[i] not in analysis_data['average.json']:
                analysis_data['average.json'][dataset_list[i]] = list()
            analysis_data['average.json'][dataset_list[i]].append(encoded_data)
    average_distance_matrix = np.zeros((len(data_map.dataset_list), len(data_map.dataset_list)))

    for i, dataset_name_a in enumerate(data_map.dataset_list):
        for j, dataset_name_b in enumerate(data_map.dataset_list):
            if i == j :
                continue
            average_distance = 0
            encoded_a = analysis_data['average.json'][dataset_name_a]
            encoded_b = analysis_data['average.json'][dataset_name_b]
            for k in range(len(analysis_data) - 2):
                average_distance += euclidean_distances(encoded_a[k].reshape(1,-1), encoded_b[k].reshape(1,-1))[0][0]
            average_distance_matrix[i][j] = average_distance
    analysis_data['average.json'] = average_distance_matrix
    model, cluster_result, _, _ = analysis_model.analyze(analysis_data['average.json'], 'average', analysis_args,encode=False, dataset_list=list(data_map.dataset_list))
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
            plt_file = os.path.join(analysis_args.analysis_result_dir, analysis_args.analysis_encode_method+'_'+method+'_'+ k.split('.')[0]+'.png')
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
        base_model = joblib.load('log/baseline/model/baseline_trust_'+data_args.data_type+'.c')
        model_list['base'] = base_model
        base_model = joblib.load('log/baseline/model/baseline_source_'+data_args.data_type+'.c')
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
        analysis_result = {k:v for k,v in analysis_result}

        result_path = os.path.join(analysis_args.analysis_result_dir, analysis_args.graph_distance)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        result_file = os.path.join(result_path,analysis_args.analysis_encode_method+'_'+method+'_'+analysis_args.graph_kernel+'.txt')
        with open(result_file, mode='w',encoding='utf8') as fp:
            for k, v in analysis_result.items():
                fp.write(k+' : '+str(v)+'\n')
    return analysis_result

def label_score_predict(
    misc_args: MiscArgument,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    adapter_args: AdapterArguments,
) -> Dict:
    dataset_map = TrustMap()
    
    data_type = list()
    dataset = data_args.dataset

    if dataset in dataset_map.position_list:
        data_type.append('position')
    data_type.append('dataset')
    if dataset in ['vanilla']:
        data_type = ['dataset', 'position']


    masked_sentence_dict: Dict = dict()
    masked_sentence_list: List = list()
    origianl_sentence_list: List = list()
    batched_masked_sentence_list: List = list()
    filter_origianl_sentence_list: List = list()

    masked_sentence_file_list: List = [os.path.join(os.path.join(os.path.join(data_args.data_dir,file_path),'article'),'en.valid') for file_path in os.listdir(data_args.data_dir)]
    for masked_sentence_file in masked_sentence_file_list:
        with open(masked_sentence_file, mode='r', encoding='utf8') as fp:
            for line in fp.readlines():
                origianl_sentence_list.append(line.strip())

    for index, masked_sentence in enumerate(origianl_sentence_list):
        sentence_list = list()
        original_sentence = masked_sentence.split(' ')
        for i, word in enumerate(original_sentence):
            raw_word = word
            original_sentence[i] = '[MASK]'
            masked_setence = ' '.join(original_sentence)
            sentence_list.append(masked_setence)
            original_sentence[i] = raw_word
        if len(sentence_list) > 1:
            masked_sentence_list.extend(sentence_list)
            filter_origianl_sentence_list.append(masked_sentence)
            for sentence in sentence_list:
                masked_sentence_dict[sentence] = index
    
    batch_size = 50
    index = 0
    while (index < len(masked_sentence_list)):
        batched_masked_sentence_list.append(masked_sentence_list[index:index+batch_size])
        index += batch_size


    model = MLMAdapterModel(model_args, data_args, training_args, adapter_args)
    word_set = set()

    log_dir = os.path.join(misc_args.log_dir, data_args.data_type)+'_'+data_args.data_dir.split('_')[1].split('/')[0]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(os.path.join(log_dir, 'json'))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, data_args.dataset+'.json')

    results = dict()
    for batch_sentence in tqdm(batched_masked_sentence_list):
        result = model.predict(batch_sentence)
        results.update(result)



    record_dict = dict()

    for sentence, items in results.items():
        try:
            original_sentence = filter_origianl_sentence_list[masked_sentence_dict[sentence]]
            if original_sentence not in record_dict:
                record_dict[original_sentence] = {'sentence':original_sentence,'word':dict()}

            masked_index = sentence.split(' ').index('[MASK]')
            record_dict[original_sentence]['word'][masked_index] = dict()
            for item in items:
                record_dict[original_sentence]['word'][masked_index][item["token_str"]] = str(round(item["score"], 3))
                word_set.add(item["token_str"])
        except:
            continue

    with open(log_file, mode='a', encoding='utf8') as fp:
        for _, item in record_dict.items():
            fp.write(json.dumps(item, ensure_ascii=False)+'\n')  




    # for original_sentence, sentence_list in tqdm(masked_sentence_dict.items()):
    #     result_dict = {'sentence':original_sentence,'word':dict()}
    #     item_dict = model.predict(sentence_list)

    #     for sentence, results in item_dict.items():
    #         masked_index = sentence.split(' ').index('[MASK]')
    #         result_dict['word'][masked_index] = dict()
    #         for result in results:
    #             result_dict['word'][masked_index][result["token_str"]] = str(round(result["score"], 3))
    #             word_set.add(result["token_str"])

    #     with open(log_file, mode='a', encoding='utf8') as fp:
    #         fp.write(json.dumps(result_dict, ensure_ascii=False)+'\n')  

    dict_file = os.path.join(os.path.join(log_dir, 'dict'), 'word_set.txt')
    if not os.path.exists(os.path.join(log_dir, 'dict')):
        os.makedirs(os.path.join(log_dir, 'dict'))
    if os.path.exists(dict_file):
        with open(dict_file,mode='r',encoding='utf8') as fp:
            for line in fp.readlines():
                word_set.add(line.strip())   
    with open(dict_file, mode='w',encoding='utf8') as fp:
        for token in word_set:
            fp.write(token+'\n')

def label_score_analysis(
    misc_args: MiscArgument,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    analysis_args: AnalysisArguments
) -> Dict:
    data_map = ArticleMap () if data_args.data_type == 'article' else TwitterMap()
    analysis_result = dict()
    model_list = dict()
    analysis_data = dict()
    sentence_position_data = dict()

    print("Load data")

    analysis_data_temp = get_label_data(analysis_args, data_args)
    index = 0
    for k, item in tqdm(analysis_data_temp.items()):
        for position, v in item.items():
            sentence_position_data[index] = {'sentence':k, 'position':position, 'word':k.split(' ')[int(position)]}
            analysis_data[index] = dict()
            for dataset in data_map.dataset_list:
                analysis_data[index][dataset] = v[dataset]
            index += 1
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

    print("Build cluster")
    method = str()
    if analysis_args.analysis_compare_method == 'cluster':
        method = analysis_args.analysis_cluster_method
        analysis_model = ClusterAnalysis(misc_args, model_args, data_args, training_args, analysis_args)
    elif analysis_args.analysis_compare_method == 'distance':
        method = analysis_args.analysis_distance_method
        analysis_model = DistanceAnalysis(misc_args, model_args, data_args, training_args, analysis_args)
    for k, v in tqdm(analysis_data.items()):
        if k == 'media_average' or k == 'concatenate':
            continue
        try:
            model, cluster_result, dataset_list, encoded_list = analysis_model.analyze(v, str(k), analysis_args, keep_result=False)
            analysis_result[k] = cluster_result
            model_list[k] = model
            for i, encoded_data in enumerate(encoded_list):
                if dataset_list[i] not in analysis_data['media_average']:
                    analysis_data['media_average'][dataset_list[i]] = list()
                analysis_data['media_average'][dataset_list[i]].append(encoded_data)
        except ValueError:
            continue
    average_distance_matrix = np.zeros((len(data_map.dataset_list), len(data_map.dataset_list)))


    conclusion = dict()
    # for k, v in analysis_result.items():
    #     analysis_file = os.path.join(analysis_args.analysis_result_dir, k.split('.')[0])
    #     with open(analysis_file, mode='a',encoding='utf8') as fp:
    #         fp.write(json.dumps({'encode': analysis_args.analysis_encode_method,'method':method, 'result':v},ensure_ascii=False)+'\n')
    #     for country, distance in v.items():
    #         if country not in conclusion:
    #             conclusion[country] = dict()
    #         conclusion[country][k] = distance

    print("Compare distance")
    if analysis_args.analysis_compare_method == 'distance':
        for k, v in analysis_result.items():
            label_list, data = v
            _draw_heatmap(data, label_list, label_list)
            # data = pd.DataFrame(v,columns=k,index=k)
            # sns.heatmap(data)
            plt_file = os.path.join(analysis_args.analysis_result_dir, analysis_args.analysis_encode_method+'_'+method+'_'+ k.split('.')[0]+'.png')
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
        base_model = joblib.load('log/baseline/model/baseline_trust_'+data_args.data_type+'.c')
        model_list['base'] = base_model
        base_model = joblib.load('log/baseline/model/baseline_source_'+data_args.data_type+'.c')
        model_list['distance_base'] = base_model
        cluster_compare = ClusterCompare(misc_args, analysis_args)
        analysis_result = cluster_compare.compare(model_list)

        print("Combine cluster")
        for i, dataset_name_a in enumerate(tqdm(data_map.dataset_list)):
            for j, dataset_name_b in enumerate(data_map.dataset_list):
                if i == j or average_distance_matrix[i][j] != 0:
                    continue
                average_distance = 0
                encoded_a = analysis_data['media_average'][dataset_name_a]
                encoded_b = analysis_data['media_average'][dataset_name_b]
                for k in range(len(encoded_a)):
                    if k in analysis_result and (analysis_result[k] < analysis_args.analysis_threshold or analysis_args.analysis_threshold == -1):
                        average_distance += euclidean_distances(encoded_a[k].reshape(1,-1), encoded_b[k].reshape(1,-1))[0][0]
                average_distance_matrix[i][j] = average_distance / len(encoded_a)
                average_distance_matrix[j][i] = average_distance / len(encoded_a)
        analysis_data['media_average'] = average_distance_matrix
        print("Combine cluster analyze")
        model, cluster_result, _, _ = analysis_model.analyze(analysis_data['media_average'], 'media_average', analysis_args, encode=False, dataset_list=list(data_map.dataset_list))
        model_list['media_average'] = model

        cluster_average = list()
        for _, v in analysis_result.items():
            if v < analysis_args.analysis_threshold or analysis_args.analysis_threshold == -1:
                cluster_average.append(v)

        analysis_result = cluster_compare.compare(model_list)
        analysis_result['cluster_average'] = np.mean(cluster_average)
        analysis_result = sorted(analysis_result.items(), key=lambda x: x[1])
        sentence_position_data['media_average'] = {'sentence':'media_average','position':-1,'word':'media_average'}
        sentence_position_data['cluster_average'] = {'sentence':'cluster_average','position':-1,'word':'cluster_average'}
        sentence_position_data['distance_base'] = {'sentence':'distance_base','position':-1,'word':'distance_base'}

        result = dict()
        average_distance = dict()
        for k, v in tqdm(analysis_result):
            sentence = sentence_position_data[k]['sentence']
            position = sentence_position_data[k]['position']
            word = sentence_position_data[k]['word']
            if sentence not in result:
                average_distance[sentence] = list()
                result[sentence] = dict()
            result[sentence][position] = (v, word)
            average_distance[sentence].append(v)
        
        for sentence, average_distance in average_distance.items():
            result[sentence]['-1'] = (np.mean(average_distance),'sentence_average')

        sentence_list = list(result.keys())
        analysis_result = {k:{'score':v, 'sentence':sentence_list.index(sentence_position_data[k]['sentence'])+1, 'position':sentence_position_data[k]['position'],'word':sentence_position_data[k]['word']} for k,v in analysis_result}

        result_path = os.path.join(analysis_args.analysis_result_dir, analysis_args.graph_distance)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        result_file = os.path.join(result_path,analysis_args.analysis_encode_method+'_'+method+'_'+analysis_args.graph_kernel+'_sort.json')
        with open(result_file, mode='w',encoding='utf8') as fp:
            for k, v in analysis_result.items():
                fp.write(json.dumps(v,ensure_ascii=False)+'\n')

        result_file = os.path.join(result_path,analysis_args.analysis_encode_method+'_'+method+'_'+analysis_args.graph_kernel+'_sentence.json')
        with open(result_file, mode='w',encoding='utf8') as fp:
            for k, v in result.items():
                v['sentence'] = k
                fp.write(json.dumps(v,ensure_ascii=False)+'\n')
    print("Analysis finish")
    return analysis_result

def train_mask_score_model(model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    analysis_args: AnalysisArguments,
    adapter_args: AdapterArguments) -> None:

    model = NERModel(model_args, data_args, training_args, adapter_args)

    train_dataset, eval_dataset = get_mask_score_data(analysis_args,data_args,model.tokenizer)
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