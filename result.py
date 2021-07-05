import os

result_dir = "/data/xiaobo/media-position"

topic_list = ["obamacare"]
loss_list = ["mlm"]
augmentation_list = ["duplicate"]

result = dict()

for topic in topic_list:
    result_path = os.path.join(result_dir, 'model_'+topic)
    model_list = os.listdir(result_path)
    model_list.sort()
    for model in model_list:
        for loss_type in loss_list:
            if loss_type not in result:
                result[loss_type] = dict()
            for augmentation in augmentation_list:
                if augmentation not in result[loss_type]:
                    result[loss_type][augmentation] = dict()
                result_dir = os.path.join(os.path.join(os.path.join(result_path,model),loss_type),augmentation)
                multi_number_list = os.listdir(result_dir)
                for multi_numer in multi_number_list:
                    if multi_numer not in result[loss_type][augmentation]:
                        result[loss_type][augmentation][multi_numer] = list()
                    result_file = os.path.join(os.path.join(os.path.join(result_dir,multi_numer), 'en'),'eval_results_lm.txt')
                    with open(result_file,mode='r',encoding='utf8') as fp:
                        for line in fp.readlines():
                            r = float(line.strip().split(' = ')[-1])
                    result[loss_type][augmentation][multi_numer].append(str(round(r,3)))

    with open('./result.csv',mode='w',encoding='utf8') as fp:
        item = 'loss,augmentation,multi_number,'+','.join(model_list)
        fp.write(item+'\n')
        item = str()
        for loss_type, loss_result in result.items():
            for augmetation_type, augmentation_result in loss_result.items():
                for multi_numer, result_list in augmentation_result.items():
                    item = loss_type+','+augmetation_type+','+multi_numer+','
                    item = item +','.join(result_list)
                    fp.write(item+'\n')


            


