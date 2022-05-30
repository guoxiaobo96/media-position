topic_list = ['climate-change', 'corporate-tax',
              'drug-policy', 'gay-marriage', 'obamacare']
# label_method_list = ['question-choice','question','declarative-choice','declarative','association-choice','association']
# label_method_list = ['bigram_outer']
label_method_list = ['declarative', 'declarative-choice', 'question',
                     'question-choice', 'association', 'association-choice', 'bigram_outer']
prob_method_list = ['absolute', 'general-relative', 'media-relative']
model_list = ['roberta-base', 'bert-base-uncased', 'bert-base-cased']
# augmentation_method_list = ['manual_clean','no_augmentation']
augmentation_method_list = ['manual_clean']
token_chosen_method_list = ['maxposi', 'manual']


def get_result_in_domain():
    import os
    import json

    for model in model_list:
        for topic in ['corporate-tax', 'drug-policy', 'gay-marriage', 'obamacare']:
            data = {'SoA-t': dict(), 'SoA-s': dict(), 'MBR': dict()}
            record_file = os.path.join(os.path.join(os.path.join(
                "./analysis", model, topic, topic, "42", "mlm", "record")))
            with open(record_file, mode='r', encoding='utf8') as fp:
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
                        data[ground_truth][label_method][prob_method][toekn_chosen_method] = dict(
                        )
                    if augmentation_method not in data[ground_truth][label_method][prob_method][toekn_chosen_method]:
                        data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method] = dict(
                        )
                    if word_only not in data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method]:
                        data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only] = list()
                    data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only] = str(
                        float(performance))
            for ground_truth, _ in data.items():
                item = ""
                with open('{}.in_domain.csv'.format(ground_truth), mode='a', encoding='utf8') as fp:
                    for label_method in label_method_list:
                        for prob_method in prob_method_list:
                            for token_chosen_method in token_chosen_method_list:
                                for aug_method in augmentation_method_list:
                                    item = item + "," + \
                                        data[ground_truth][label_method][prob_method][token_chosen_method][aug_method]['True']
                        item += ','
                    fp.write(item+'\n')


def get_result_out_domain():
    import os
    import json
    import numpy as np

    for model in model_list:
        for target_topic in ['climate-change', 'corporate-tax', 'drug-policy', 'gay-marriage', 'obamacare']:
            data = {'SoA-t': dict(), 'SoA-s': dict(), 'MBR': dict()}
            for topic in ['climate-change', 'corporate-tax', 'drug-policy', 'gay-marriage', 'obamacare']:
                if target_topic == topic:
                    continue
                record_file = os.path.join(os.path.join(os.path.join(
                    "./analysis", model, topic, target_topic, "42", "mlm", "record")))
                with open(record_file, mode='r', encoding='utf8') as fp:
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
                            data[ground_truth][label_method][prob_method][toekn_chosen_method] = dict(
                            )
                        if augmentation_method not in data[ground_truth][label_method][prob_method][toekn_chosen_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method] = dict(
                            )
                        if word_only not in data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only] = list(
                            )
                        data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only].append(
                            (float(performance)))
            for ground_truth, _ in data.items():
                item = ""
                with open('{}.out_domain.csv'.format(ground_truth), mode='a', encoding='utf8') as fp:
                    # for label_method in ['question-tf','question-yn','declarative-tf','declarative-yn']:
                    for label_method in label_method_list:
                        for prob_method in prob_method_list:
                            for token_chosen_method in token_chosen_method_list:
                                for aug_method in augmentation_method_list:
                                    item = item + "," + \
                                        str(round(np.mean(
                                            data[ground_truth][label_method][prob_method][token_chosen_method][aug_method]['True']), 3))
                        item += ','
                    fp.write(item+'\n')


def get_result_list():
    import os
    import json
    import numpy as np

    for model in model_list:
        for target_topic in ['climate-change', 'corporate-tax', 'drug-policy', 'gay-marriage', 'obamacare']:
            data = {'SoA-t': dict(), 'SoA-s': dict(), 'MBR': dict()}
            for topic in ['climate-change', 'corporate-tax', 'drug-policy', 'gay-marriage', 'obamacare']:
                record_file = os.path.join(os.path.join(os.path.join(
                    "./analysis", model, topic, target_topic, "42", "mlm", "record")))
                with open(record_file, mode='r', encoding='utf8') as fp:
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
                            data[ground_truth][label_method][prob_method][toekn_chosen_method] = dict(
                            )
                        if augmentation_method not in data[ground_truth][label_method][prob_method][toekn_chosen_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method] = dict(
                            )
                        if word_only not in data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only] = list(
                            )
                        data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only].append(
                            (str(round(float(performance), 3))))
            for ground_truth, _ in data.items():
                item = ""
                with open('{}.list.csv'.format(ground_truth), mode='a', encoding='utf8') as fp:
                    for label_method in label_method_list:
                        for prob_method in prob_method_list:
                            for token_chosen_method in token_chosen_method_list:
                                for aug_method in augmentation_method_list:
                                    item = item + \
                                        ",".join(data[ground_truth][label_method][prob_method]
                                                 [token_chosen_method][aug_method]['True']) + ",,"
                    fp.write(item+'\n')


def get_result():
    import os
    import json
    import numpy as np

    for model in model_list:
        data = {'SoA-t': dict(), 'SoA-s': dict(), 'MBR': dict()}
        for topic in topic_list:
            record_file = os.path.join(os.path.join(os.path.join(
                "./analysis", model, topic, topic, "42", "mlm", "record")))
            with open(record_file, mode='r', encoding='utf8') as fp:
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
                        data[ground_truth][label_method][prob_method][toekn_chosen_method] = dict(
                        )
                    if augmentation_method not in data[ground_truth][label_method][prob_method][toekn_chosen_method]:
                        data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method] = dict(
                        )
                    if word_only not in data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method]:
                        data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only] = list()
                    data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only].append(
                        float(performance))
        for ground_truth, _ in data.items():
            with open('{}.conclusion.csv'.format(ground_truth), mode='a', encoding='utf8') as fp:
                for label_method in label_method_list:
                    for prob_method in prob_method_list:
                        for token_chosen_method in token_chosen_method_list:
                            for aug_method in augmentation_method_list:
                                item = model + "," + label_method+"," + prob_method+","+token_chosen_method+"," + aug_method+"," + \
                                    str(round(np.mean(
                                        data[ground_truth][label_method][prob_method][token_chosen_method][aug_method]['True']), 3))
                                fp.write(item+'\n')


def get_paper_result():
    import os
    import json
    import numpy as np

    for model in model_list:
        data = {'SoA-t': dict(), 'SoA-s': dict(), 'MBR': dict()}
        # data = {'human': dict()}
        for topic in topic_list:
            record_file = os.path.join(os.path.join(os.path.join(
                "./analysis", model, topic, topic, "42", "mlm", "record")))
            with open(record_file, mode='r', encoding='utf8') as fp:
                for line in fp.readlines():
                    item = json.loads(line.strip())
                    ground_truth = item['ground_truth']
                    label_method = item['label method']
                    prob_method = item['prob method']
                    toekn_chosen_method = item['token chosen method']
                    word_only = item['word only']
                    augmentation_method = item['augmentation_method']
                    performance = item['performance']
                    try:
                        if label_method not in data[ground_truth]:
                            data[ground_truth][label_method] = dict()
                        if prob_method not in data[ground_truth][label_method]:
                            data[ground_truth][label_method][prob_method] = dict()
                        if toekn_chosen_method not in data[ground_truth][label_method][prob_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method] = dict(
                            )
                        if augmentation_method not in data[ground_truth][label_method][prob_method][toekn_chosen_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method] = dict(
                            )
                        if word_only not in data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method]:
                            data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only] = list(
                            )
                        data[ground_truth][label_method][prob_method][toekn_chosen_method][augmentation_method][word_only].append(
                            float(performance))
                    except:
                        continue
        for ground_truth, _ in data.items():
            with open('{}.conclusion.csv'.format(ground_truth), mode='a', encoding='utf8') as fp:
                item = ",".join(label_method_list)
                fp.write(item+'\n')
                for prob_method in prob_method_list:
                    item = ""
                    for label_method in label_method_list:
                        for token_chosen_method in token_chosen_method_list:
                            for aug_method in augmentation_method_list:
                                try:
                                    item = item + "," + str(round(np.mean(data[ground_truth][label_method][prob_method][token_chosen_method][aug_method]['True']), 2))+"(" + str(
                                        round(np.std(data[ground_truth][label_method][prob_method][token_chosen_method][aug_method]['True'],ddof=1), 2))+")"

                                except:
                                    continue
                    fp.write(item[1:]+'\n')


if __name__ == '__main__':
    get_paper_result()
    # get_result()
    # get_result_in_domain()
    # get_result_out_domain()
    # get_result_list()
