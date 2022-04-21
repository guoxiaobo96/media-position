label_method_list = ['question-choice','question','declarative-choice','declarative']
prob_method_list =  ['absolute','media-relative','general-relative']
token_chosen_method_list = ['manual']

def get_res():
    import os
    import json

    for model in ['roberta-base','bert-base-uncased']:
        for topic in ['climate-change','corporate-tax','drug-policy','gay-marriage','obamacare']:
            data = {'SoA-t':dict(),'SoA-s':dict(),'MBR':dict()}
            record_file = os.path.join(os.path.join(os.path.join("./analysis",model,topic,topic,"42","mlm","record")))
            # record_file = "./analysis\{}\{}\{}\\42\\mlm\\record".format(model,topic,topic)
            record_file_t = r"./analysis\roberta-base\climate-change\climate-change\42\mlm\record"
            with open(record_file,mode='r',encoding='utf8') as fp:
                for line in fp.readlines():
                    item = json.loads(line.strip())
                    ground_truth = item['ground_truth']
                    label_method = item['label method']
                    prob_method = item['prob method']
                    toekn_chosen_method = item['token chosen method']
                    word_only = item['word only']
                    performance = item['performance']
                    if label_method not in data[ground_truth]:
                        data[ground_truth][label_method] = dict()
                    if prob_method not in data[ground_truth][label_method]:
                        data[ground_truth][label_method][prob_method] = dict()
                    if toekn_chosen_method not in data[ground_truth][label_method][prob_method]:
                        data[ground_truth][label_method][prob_method][toekn_chosen_method] = dict()
                    data[ground_truth][label_method][prob_method][toekn_chosen_method][word_only] = str(float(performance))
            for ground_truth, ground_trunth_data in data.items():
                item = ""
                with open('{}.csv'.format(ground_truth),mode='a',encoding='utf8') as fp:
                    for label_method in ['manual']:
                        for prob_method in ['absolute','media-relative','general-relative']:
                            # if prob_method == 'absolute':
                            #     token_chosen_method_list = ['maxposi']
                            # else:
                            #     token_chosen_method_list = ['manual']
                            token_chosen_method_list = ['manual']
                            for token_chosen_method in token_chosen_method_list:
                                for word_only in ['True','False']:
                                    item = item + ","+data[ground_truth][label_method][prob_method][token_chosen_method][word_only]
                    fp.write(item+'\n')

def get_result_in_domain():
    import os
    import json

    for model in ['roberta-base','bert-base-uncased']:
        for topic in ['climate-change','corporate-tax','drug-policy','gay-marriage','obamacare']:
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
                    performance = item['performance']
                    if label_method not in data[ground_truth]:
                        data[ground_truth][label_method] = dict()
                    if prob_method not in data[ground_truth][label_method]:
                        data[ground_truth][label_method][prob_method] = dict()
                    if toekn_chosen_method not in data[ground_truth][label_method][prob_method]:
                        data[ground_truth][label_method][prob_method][toekn_chosen_method] = dict()
                    data[ground_truth][label_method][prob_method][toekn_chosen_method][word_only] = str(float(performance))
            for ground_truth, _ in data.items():
                item = ""
                with open('{}.in_domain.csv'.format(ground_truth),mode='a',encoding='utf8') as fp:
                    for label_method in label_method_list:
                        for prob_method in prob_method_list:
                            for token_chosen_method in token_chosen_method_list:
                                item = item + ","+data[ground_truth][label_method][prob_method][token_chosen_method]['True']
                        item +=','
                    fp.write(item+'\n')

def get_result_out_domain():
    import os
    import json
    import numpy as np

    for model in ['roberta-base','bert-base-uncased']:
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
                        performance = item['performance']
                        if label_method not in data[ground_truth]:
                            data[ground_truth][label_method] = dict()
                        if prob_method not in data[ground_truth][label_method]:
                            data[ground_truth][label_method][prob_method] = dict()
                        if toekn_chosen_method not in data[ground_truth][label_method][prob_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method] = dict()
                        if word_only not in data[ground_truth][label_method][prob_method][toekn_chosen_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method][word_only] = list()
                        data[ground_truth][label_method][prob_method][toekn_chosen_method][word_only].append((float(performance)))
            for ground_truth, _ in data.items():
                item = ""
                with open('{}.out_domain.csv'.format(ground_truth),mode='a',encoding='utf8') as fp:
                    # for label_method in ['question-tf','question-yn','declarative-tf','declarative-yn']:
                    for label_method in label_method_list:
                        for prob_method in prob_method_list:
                            for token_chosen_method in token_chosen_method_list:
                                item = item + ","+str(round(np.mean(data[ground_truth][label_method][prob_method][token_chosen_method]['True']),3))
                        item +=','
                    fp.write(item+'\n')

def get_result_list():
    import os
    import json
    import numpy as np

    for model in ['roberta-base','bert-base-uncased']:
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
                        performance = item['performance']
                        if label_method not in data[ground_truth]:
                            data[ground_truth][label_method] = dict()
                        if prob_method not in data[ground_truth][label_method]:
                            data[ground_truth][label_method][prob_method] = dict()
                        if toekn_chosen_method not in data[ground_truth][label_method][prob_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method] = dict()
                        if word_only not in data[ground_truth][label_method][prob_method][toekn_chosen_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method][word_only] = list()
                        data[ground_truth][label_method][prob_method][toekn_chosen_method][word_only].append((str(round(float(performance),3))))
            for ground_truth, _ in data.items():
                item = ""
                with open('{}.list.csv'.format(ground_truth),mode='a',encoding='utf8') as fp:
                    for label_method in label_method_list:
                        for prob_method in prob_method_list:
                            for token_chosen_method in token_chosen_method_list:
                                item = item + ",".join(data[ground_truth][label_method][prob_method][token_chosen_method]['True']) +",,"
                    fp.write(item+'\n')
if __name__ == '__main__':
    get_result_in_domain()
    get_result_out_domain()
    get_result_list()