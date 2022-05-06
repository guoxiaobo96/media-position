topic_list=['corporate-tax','drug-policy','gay-marriage','obamacare']
# label_method_list = ['question-choice','question','declarative-choice','declarative','association-choice','association']
label_method_list = ['bigram_outer']
prob_method_list =  ['absolute','media-relative','general-relative']
model_list =  ['roberta-base','bert-base-uncased','bert-base-cased']
augmentation_method_list = ['manual_clean','no_augmentation']
# token_chosen_method_list = ['manual']
token_chosen_method_list = ['maxposi']


def get_result_in_domain():
    import os
    import json

    for model in model_list:
        for topic in ['corporate-tax','drug-policy','gay-marriage','obamacare']:
            data = {'SoA-t':dict(),'SoA-s':dict(),'MBR':dict()}
            record_file = os.path.join(os.path.join(os.path.join("./analysis",model,topic,topic,"42","mlm","record")))
            with open(record_file,mode='r',encoding='utf8') as fp:
                for line in fp.readlines():
                    item = json.loads(line.strip())
                    ground_truth = item['ground_truth']
                    label_method = item['label method']
                    prob_method = item['prob method']
                    toekn_chosen_method = item['token chosen method']
                    word_only = item['word only']
                    augmentation_method = item['augmentation_method']
                    performance = item['performance']
                    if label_method not in data[ground_truth]:
                        data[ground_truth][label_method] = dict()
                    if prob_method not in data[ground_truth][label_method]:
                        data[ground_truth][label_method][prob_method] = dict()
                    if toekn_chosen_method not in data[ground_truth][label_method][prob_method]:
                        data[ground_truth][label_method][prob_method][toekn_chosen_method] = dict()
                    if augmentation_method not in data[ground_truth][label_method][prob_method][toekn_chosen_method]:
                        data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method] = dict()
                    if word_only not in data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only] = list()
                    data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only] = str(float(performance))
            for ground_truth, _ in data.items():
                item = ""
                with open('{}.in_domain.csv'.format(ground_truth),mode='a',encoding='utf8') as fp:
                    for label_method in label_method_list:
                        for prob_method in prob_method_list:
                            for token_chosen_method in token_chosen_method_list:
                                for aug_method in augmentation_method_list:
                                    item = item + ","+data[ground_truth][label_method][prob_method][token_chosen_method][aug_method]['True']
                        item +=','
                    fp.write(item+'\n')

def get_result_out_domain():
    import os
    import json
    import numpy as np

    for model in model_list:
        for target_topic in ['climate-change','corporate-tax','drug-policy','gay-marriage','obamacare']:
            data = {'SoA-t':dict(),'SoA-s':dict(),'MBR':dict()}
            for topic in ['climate-change','corporate-tax','drug-policy','gay-marriage','obamacare']:
                if target_topic == topic:
                    continue
                record_file = os.path.join(os.path.join(os.path.join("./analysis",model,topic,target_topic,"42","mlm","record")))
                with open(record_file,mode='r',encoding='utf8') as fp:
                    for line in fp.readlines():
                        item = json.loads(line.strip())
                        ground_truth = item['ground_truth']
                        label_method = item['label method']
                        prob_method = item['prob method']
                        toekn_chosen_method = item['token chosen method']
                        word_only = item['word only']
                        augmentation_method = item['augmentation_method']
                        performance = item['performance']
                        if label_method not in data[ground_truth]:
                            data[ground_truth][label_method] = dict()
                        if prob_method not in data[ground_truth][label_method]:
                            data[ground_truth][label_method][prob_method] = dict()
                        if toekn_chosen_method not in data[ground_truth][label_method][prob_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method] = dict()
                        if augmentation_method not in data[ground_truth][label_method][prob_method][toekn_chosen_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method] = dict()
                        if word_only not in data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method]:
                                data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only] = list()
                        data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only].append((float(performance)))
            for ground_truth, _ in data.items():
                item = ""
                with open('{}.out_domain.csv'.format(ground_truth),mode='a',encoding='utf8') as fp:
                    # for label_method in ['question-tf','question-yn','declarative-tf','declarative-yn']:
                    for label_method in label_method_list:
                        for prob_method in prob_method_list:
                            for token_chosen_method in token_chosen_method_list:
                                for aug_method in augmentation_method_list:
                                    item = item + ","+str(round(np.mean(data[ground_truth][label_method][prob_method][token_chosen_method][aug_method]['True']),3))
                        item +=','
                    fp.write(item+'\n')

def get_result_list():
    import os
    import json
    import numpy as np

    for model in model_list:
        for target_topic in ['climate-change','corporate-tax','drug-policy','gay-marriage','obamacare']:
            data = {'SoA-t':dict(),'SoA-s':dict(),'MBR':dict()}
            for topic in ['climate-change','corporate-tax','drug-policy','gay-marriage','obamacare']:
                record_file = os.path.join(os.path.join(os.path.join("./analysis",model,topic,target_topic,"42","mlm","record")))
                with open(record_file,mode='r',encoding='utf8') as fp:
                    for line in fp.readlines():
                        item = json.loads(line.strip())
                        ground_truth = item['ground_truth']
                        label_method = item['label method']
                        prob_method = item['prob method']
                        toekn_chosen_method = item['token chosen method']
                        word_only = item['word only']
                        augmentation_method = item['augmentation_method']
                        performance = item['performance']
                        if label_method not in data[ground_truth]:
                            data[ground_truth][label_method] = dict()
                        if prob_method not in data[ground_truth][label_method]:
                            data[ground_truth][label_method][prob_method] = dict()
                        if toekn_chosen_method not in data[ground_truth][label_method][prob_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method] = dict()
                        if augmentation_method not in data[ground_truth][label_method][prob_method][toekn_chosen_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method] = dict()
                        if word_only not in data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method]:
                                data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only] = list()
                        data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only].append((str(round(float(performance),3))))
            for ground_truth, _ in data.items():
                item = ""
                with open('{}.list.csv'.format(ground_truth),mode='a',encoding='utf8') as fp:
                    for label_method in label_method_list:
                        for prob_method in prob_method_list:
                            for token_chosen_method in token_chosen_method_list:
                                for aug_method in augmentation_method_list:
                                    item = item + ",".join(data[ground_truth][label_method][prob_method][token_chosen_method][aug_method]['True']) +",,"
                    fp.write(item+'\n')

def get_result():
    import os
    import json
    import numpy as np

    for model in model_list:
        data = {'SoA-t':dict(),'SoA-s':dict(),'MBR':dict()}
        for topic in topic_list:
            record_file = os.path.join(os.path.join(os.path.join("./analysis",model,topic,topic,"42","mlm","record")))
            with open(record_file,mode='r',encoding='utf8') as fp:
                for line in fp.readlines():
                    item = json.loads(line.strip())
                    ground_truth = item['ground_truth']
                    label_method = item['label method']
                    prob_method = item['prob method']
                    toekn_chosen_method = item['token chosen method']
                    word_only = item['word only']
                    augmentation_method = item['augmentation_method']
                    performance = item['performance']
                    if label_method not in data[ground_truth]:
                        data[ground_truth][label_method] = dict()
                    if prob_method not in data[ground_truth][label_method]:
                        data[ground_truth][label_method][prob_method] = dict()
                    if toekn_chosen_method not in data[ground_truth][label_method][prob_method]:
                        data[ground_truth][label_method][prob_method][toekn_chosen_method] = dict()
                    if augmentation_method not in data[ground_truth][label_method][prob_method][toekn_chosen_method]:
                        data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method] = dict()
                    if word_only not in data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only] = list()
                    data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only].append(float(performance))
        for ground_truth, _ in data.items():
            with open('{}.conclusion.csv'.format(ground_truth),mode='a',encoding='utf8') as fp:
                for label_method in label_method_list:
                    for prob_method in prob_method_list:
                        for token_chosen_method in token_chosen_method_list:
                            for aug_method in augmentation_method_list:
                                item = model +","+ label_method+","+ prob_method+","+token_chosen_method+","+ aug_method+"," + str(round(np.mean(data[ground_truth][label_method][prob_method][token_chosen_method][aug_method]['True']),3))
                                fp.write(item+'\n')

def log_analysis():
    import os
    import pandas as pd
    import statsmodels.api as sm
    import csv
    dummy_feature_list = ['model','label_method','prob_method','aug_method']
    data = list()
    for ground_truth in ['SoA-t','SoA-s','MBR']:
        with open('{}.conclusion.csv'.format(ground_truth),mode='r',encoding='utf8') as fp:
            csv_reader = csv.reader(fp,delimiter=',')
            for row in csv_reader:
                item = row[0:3] +row[4:5]+[float(row[-1])]
                data.append(item)
        
    data = pd.DataFrame(data,columns=dummy_feature_list+['performance'])
    for model in model_list:
        print('\n' + model+'\n')
        model_data = data.loc[data['model'] == model]
        for feature in dummy_feature_list:
            if feature != 'model':
                feature_data = model_data.groupby([feature])
                print(feature_data.describe())
        for feature in dummy_feature_list:
                feature_data = data.groupby([feature])
                print(feature_data.describe())
        # y_data = data[['performance']]
        # x_data = data.drop(['performance'],axis=1)
        # dummy = pd.get_dummies(x_data[dummy_feature_list])
        # x_data.drop(dummy_feature_list, axis=1, inplace=True)
        # x_data = pd.concat([x_data, dummy], axis=1)
        # x_data = sm.add_constant(x_data, has_constant='add')
        # model = sm.Logit(y_data, x_data.astype(
        #     float)).fit(maxiter=200)
        # report_file = '{}.logReg'.format(ground_truth)
        # with open(report_file, mode='w') as fp:
        #     fp.write(model.summary().as_text())

    # label_method_list = ['bigram_outer', 'question-choice','question','declarative-choice','declarative','association-choice', 'association']
    # prob_method_list =  ['absolute','media-relative','general-relative']
    # model_list =  ['roberta-base','bert-base-uncased','bert-base-cased']
    # augmentation_method_list = ['manual_clean','no_augmentation']
    # token_chosen_method_list = ['maxposi','manual']
    dummy_feature_list = ['label_method','prob_method','model','augmentation_method','token_chosen_method']
    


if __name__ == '__main__':
    # get_result()
    log_analysis()
    # get_result_in_domain()
    # get_result_out_domain()
    # get_result_list()