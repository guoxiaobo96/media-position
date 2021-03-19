from typing import Dict, List
import json
import os

from transformers.utils.dummy_tokenizers_objects import convert_slow_tokenizer
from .config import DataArguments, MiscArgument, ModelArguments, TrainingArguments, AdapterArguments, AnalysisArguments, SourceMap, TrustMap
from .model import AdapterModel
from .data import get_dataset, get_analysis_data
from .analysis import ClusterAnalysis,DistanceAnalysis


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
    if dataset in dataset_map.dataset_to_name:
        data_type.append('dataset')
    if dataset in dataset_map.position_list:
        data_type.append('position')
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
    analysis_result = dict()
    analysis_data = get_analysis_data(analysis_args)
    method = str()
    if analysis_args.analysis_compare_method == 'cluster':
        analysis_model = ClusterAnalysis(misc_args, model_args, data_args, training_args, analysis_args)
        method = analysis_args.analysis_cluster_method
    elif analysis_args.analysis_compare_method == 'distance':
        analysis_model = DistanceAnalysis(misc_args, model_args, data_args, training_args, analysis_args)
        method = analysis_args.analysis_distance_method
    for k, v in analysis_data.items():
        analysis_result[k] = analysis_model.analyze(v, k.split('.')[0], analysis_args)
    conclusion = dict()
    for k, v in analysis_result.items():
        analysis_file = os.path.join(analysis_args.analysis_result_dir, k.split('.')[0])
        with open(analysis_file, mode='a',encoding='utf8') as fp:
            fp.write(json.dumps({'encode': analysis_args.analysis_encode_method,'method':method, 'result':v},ensure_ascii=False)+'\n')
        for country, distance in v.items():
            if country not in conclusion:
                conclusion[country] = dict()
            conclusion[country][k] = distance
    if analysis_args.analysis_compare_method == 'distance':
        with open(os.path.join(analysis_args.analysis_result_dir, analysis_args.analysis_encode_method+'_'+method+'.csv'), mode='w',encoding='utf8') as fp:
            title = 'country,'
            for i in range(len(analysis_result)):
                title = title + str(i+1)+','
            fp.write(title+'\n')
            for country, distance_list in conclusion.items():
                record = country+','
                for i in range(len(distance_list)):
                    record = record+str(distance_list[str(i+1)+'.json'])+','
                fp.write(record+'\n')

        
    return analysis_result