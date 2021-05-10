from typing import Dict, List
import json
import os
import joblib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


from .config import DataArguments, MiscArgument, ModelArguments, TrainingArguments, AdapterArguments, AnalysisArguments, SourceMap, TrustMap, TwitterMap, ArticleMap
from .model import AdapterModel
from .data import get_dataset, get_analysis_data
from .analysis import ClusterAnalysis,DistanceAnalysis,ClusterCompare


def train_adapter(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    adapter_args: AdapterArguments
) -> Dict:
    model = AdapterModel(model_args, data_args, training_args, adapter_args)
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


def predict_adapter(
    misc_args: MiscArgument,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    adapter_args: AdapterArguments,
) -> Dict:
    # dataset_map = (
    #     SourceMap() if misc_args.target=='source' else TrustMap()
    # )
    dataset_map = TrustMap()
    
    data_type = list()
    dataset = data_args.dataset
    # if dataset in dataset_map.dataset_to_name:
    #     data_type.append('dataset')
    if dataset in dataset_map.position_list:
        data_type.append('position')
    data_type.append('dataset')
    if dataset in ['vanilla']:
        data_type = ['dataset', 'position']
        
    model = AdapterModel(model_args, data_args, training_args, adapter_args)
    masked_sentence_file: str = './masked_sentence'
    masked_sentence_list: List = list()
    log_dir = os.path.join(misc_args.log_dir, data_args.data_type)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(masked_sentence_file, mode='r', encoding='utf8') as fp:
        for line in fp.readlines():
            masked_sentence_list.append(line.strip())
    result_dict = model.predict(masked_sentence_list)
    dict_set = set()
    dict_file = os.path.join(os.path.join(log_dir, 'dict'), 'word_set.txt')
    if not os.path.exists(os.path.join(log_dir, 'dict')):
        os.makedirs(os.path.join(log_dir, 'dict'))
    if os.path.exists(dict_file):
        with open(dict_file,mode='r',encoding='utf8') as fp:
            for line in fp.readlines():
                dict_set.add(line.strip())

    for file_format in ['json', 'csv']:
        if not os.path.exists(os.path.join(log_dir, file_format)):
            os.makedirs(os.path.join(log_dir, file_format))
        for i, sentence in enumerate(masked_sentence_list):
            log_file = os.path.join(os.path.join(log_dir, file_format), str(i+1)+'.'+file_format)
            if not os.path.exists(log_file):
                with open(log_file, mode='w', encoding='utf8') as fp:
                    fp.write(json.dumps({"sentence": sentence},
                                        ensure_ascii=False) + '\n')
            if file_format == "json":
                record = {"dataset": dataset, "data_type":data_type, "words": {}}

                with open(log_file, mode='a', encoding='utf8') as fp:
                    results = result_dict[sentence]
                    for result in results:
                        record["words"][result["token_str"]] = str(
                            round(result["score"], 3))
                    fp.write(json.dumps(record, ensure_ascii=False)+'\n')
            elif file_format == "csv":
                tokens = dataset+','
                scores = dataset+','

                with open(log_file, mode='a',encoding='utf8') as fp:
                    results = result_dict[sentence]
                    for result in results:
                        tokens = tokens + result["token_str"]+","
                        scores = scores + str(round(result["score"],3))+","
                        dict_set.add(result["token_str"])
                    fp.write(tokens+'\n'+scores+'\n')
    with open(dict_file, mode='w',encoding='utf8') as fp:
        for token in dict_set:
            fp.write(token+'\n')


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
        base_model = joblib.load('log/baseline/model/baseline_source_'+data_args.data_type+'.c')
        model_list['base'] = base_model
        base_model = joblib.load('log/baseline/model/baseline_trust_'+data_args.data_type+'.c')
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